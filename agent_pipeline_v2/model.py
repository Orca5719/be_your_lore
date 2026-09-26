"""V2-only extension of the frozen single-request Qwen adapter."""

from __future__ import annotations

import time

from qwen_judge import MODEL, REVISION, GenerationError, QwenJudge

from .batch_llm import left_pad_rows


class V2QwenJudge(QwenJudge):
    def __init__(self, device="auto"):
        super().__init__(device)
        self.last_batch_generation = {}

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
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        started = time.perf_counter()
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
            )
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        seconds = time.perf_counter() - started
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
        self.last_batch_generation = {
            "seconds": seconds,
            "batch_size": len(rows),
            "input_tokens": batch["input_lengths"],
            "useful_input_tokens": sum(batch["input_lengths"]),
            "padded_input_tokens": batch["padded_input_tokens"],
            "padded_width": width,
            "generated_tokens": generated_tokens,
            "total_generated_tokens": sum(generated_tokens),
            "truncated_indices": truncated,
        }
        return raw_outputs


__all__ = ["MODEL", "REVISION", "GenerationError", "V2QwenJudge"]
