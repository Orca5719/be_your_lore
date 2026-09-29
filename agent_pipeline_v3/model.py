from __future__ import annotations

from transformers import StoppingCriteriaList

from agent_pipeline_v2.batch_llm import left_pad_rows
from agent_pipeline_v2.model import V2QwenJudge
from model_metrics import TokenTiming
from qwen_judge import GenerationError, MODEL, REVISION

from .profiling import FirstTokenTimer, TraceCollector


class ProfiledV2QwenJudge(V2QwenJudge):
    """The frozen v2 model adapter with generation-only profiling."""

    def __init__(self, device="auto"):
        super().__init__(device)
        self.collector = TraceCollector()

    def set_story_id(self, story_id: str | None) -> None:
        self.collector.set_story(story_id)

    def clear_traces(self) -> None:
        self.collector.clear()

    @property
    def inference_traces(self) -> list[dict]:
        return list(self.collector.records)

    def _cuda_begin(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()

    def _cuda_end(self) -> tuple[int | None, int | None]:
        if self.device != "cuda":
            return None, None
        self.torch.cuda.synchronize()
        return self.torch.cuda.max_memory_allocated(), self.torch.cuda.max_memory_reserved()

    def _generate(self, messages, max_new_tokens=768):
        self.last_generation = {}
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        count = inputs["input_ids"].shape[1]
        if count + max_new_tokens > 4096:
            raise ValueError("输入和证据超过 4096 tokens 预算，不会截断")
        self._cuda_begin()
        span = self.collector.start("extractor", 1, int(count), 0)
        token_timing = TokenTiming(clock=self.collector.clock)
        output = None
        try:
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=50,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    streamer=token_timing,
                )
            peak_allocated, peak_reserved = self._cuda_end()
            span.first_token_ready = token_timing.first
            generated_tokens = int(output.shape[1] - count)
            record = span.finish(
                generated_tokens,
                peak_allocated=peak_allocated,
                peak_reserved=peak_reserved,
            )
        except Exception as exc:
            peak_allocated, peak_reserved = self._cuda_end()
            span.first_token_ready = token_timing.first
            record = span.finish(
                0 if output is None else int(output.shape[1] - count),
                status="error",
                peak_allocated=peak_allocated,
                peak_reserved=peak_reserved,
                error=f"{type(exc).__name__}: {exc}",
            )
            self.last_generation = {"call_id": record["call_id"], "profile": record}
            raise
        decode_tokens = max(0, generated_tokens - 1)
        self.last_generation = {
            "seconds": record["total_time"],
            "input_tokens": count,
            "generated_tokens": generated_tokens,
            "ttft_ms": record["ttft_ms"],
            "decode_seconds": record["decode_time"],
            "decode_tokens": decode_tokens,
            "decode_tokens_per_second": (
                decode_tokens / record["decode_time"] if record["decode_time"] and decode_tokens else None
            ),
            "call_id": record["call_id"],
            "profile": record,
        }
        ids = output[0, count:].tolist()
        raw = self.tokenizer.decode(ids, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos or []
        if len(ids) == max_new_tokens and ids[-1] not in eos:
            raise GenerationError("生成达到上限且未结束", raw)
        return raw

    def _generate_batch(self, message_batches, max_new_tokens=768):
        if not isinstance(message_batches, list) or not message_batches:
            raise ValueError("批量消息不能为空")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens必须为正整数")
        rows = []
        for messages in message_batches:
            ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if ids and isinstance(ids[0], list):
                if len(ids) != 1:
                    raise ValueError("单条chat template产生了意外batch维度")
                ids = ids[0]
            if not isinstance(ids, list) or not ids:
                raise ValueError("chat template没有产生token")
            rows.append(ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("tokenizer缺少pad/eos token")
        batch = left_pad_rows(rows, pad_id, self.torch, self.device)
        width = batch["padded_width"]
        if width + max_new_tokens > 4096:
            raise ValueError("批量输入和输出超过4096 tokens预算，不会截断")
        self.last_batch_generation = {}
        self._cuda_begin()
        span = self.collector.start(
            "judge",
            len(rows),
            int(sum(batch["input_lengths"])),
            int(batch["padded_input_tokens"]),
        )
        timer = FirstTokenTimer(span, self.torch.cuda.synchronize if self.device == "cuda" else None)
        output = None
        try:
            with self.torch.inference_mode():
                output = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=50,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=pad_id,
                    stopping_criteria=StoppingCriteriaList([timer]),
                )
            peak_allocated, peak_reserved = self._cuda_end()
        except Exception as exc:
            peak_allocated, peak_reserved = self._cuda_end()
            record = span.finish(
                0,
                status="error",
                peak_allocated=peak_allocated,
                peak_reserved=peak_reserved,
                error=f"{type(exc).__name__}: {exc}",
            )
            self.last_batch_generation = {"call_id": record["call_id"], "profile": record}
            raise
        generated = output[:, width:]
        eos = self.model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        generated_tokens = []
        truncated = []
        raw_outputs = []
        for index, row in enumerate(generated.tolist()):
            meaningful = []
            for token in row:
                if token == pad_id and pad_id not in eos_ids:
                    break
                meaningful.append(token)
                if token in eos_ids:
                    break
            generated_tokens.append(len(meaningful))
            if len(row) >= max_new_tokens and (not meaningful or meaningful[-1] not in eos_ids):
                truncated.append(index)
            raw_outputs.append(self.tokenizer.decode(row, skip_special_tokens=True))
        record = span.finish(
            int(sum(generated_tokens)),
            peak_allocated=peak_allocated,
            peak_reserved=peak_reserved,
        )
        self.last_batch_generation = {
            "seconds": record["total_time"],
            "batch_size": len(rows),
            "input_tokens": batch["input_lengths"],
            "useful_input_tokens": sum(batch["input_lengths"]),
            "padded_input_tokens": batch["padded_input_tokens"],
            "padded_width": width,
            "generated_tokens": generated_tokens,
            "total_generated_tokens": sum(generated_tokens),
            "truncated_indices": truncated,
            "call_id": record["call_id"],
            "profile": record,
        }
        return raw_outputs


__all__ = ["MODEL", "REVISION", "GenerationError", "ProfiledV2QwenJudge"]
