"""Keep Benchmark 3 generation intact while constraining Extractor JSON syntax."""

from __future__ import annotations

from contextlib import contextmanager
import json
import time

from agent_pipeline_v3.model import ProfiledV2QwenJudge


def extractor_json_schema() -> dict:
    """The frozen Extractor v1 wire shape; semantic checks remain downstream."""
    text = {"type": "string"}
    texts = {"type": "array", "items": text}
    event = {
        "type": "object",
        "properties": {
            "actors": texts,
            "event": text,
            "mental_state": {"type": ["string", "null"]},
            "explicit": {"type": "boolean"},
            "modality": {"type": "string", "enum": ["observed", "speech", "belief", "plan", "dream", "conditional", "inferred"]},
            "conditions": texts,
            "source_ids": texts,
            "context_ids": texts,
            "check_reason": {"type": "string", "enum": ["mechanism", "knowledge_relation", "state_time_space", "consequence_support"]},
        },
        "required": ["actors", "event", "mental_state", "explicit", "modality", "conditions", "source_ids", "context_ids", "check_reason"],
        "additionalProperties": False,
    }
    ignored = {
        "type": "object",
        "properties": {"source_id": text, "reason": {"type": "string", "enum": ["routine", "process_detail"]}},
        "required": ["source_id", "reason"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "events": {"type": "array", "items": event},
            "ignored_spans": {"type": "array", "items": ignored},
            "non_event_span_ids": texts,
        },
        "required": ["events", "ignored_spans", "non_event_span_ids"],
        "additionalProperties": False,
    }


@contextmanager
def constrained_generation(model, processor_factory):
    """Inject a fresh processor into one generate call, even if it raises."""
    original = model.generate

    def generate(*args, **kwargs):
        if "logits_processor" in kwargs:
            raise ValueError("Extractor generation already has a logits processor")
        return original(*args, **kwargs, logits_processor=[processor_factory()])

    model.generate = generate
    try:
        yield
    finally:
        model.generate = original


class TorchCompatibleProcessor:
    """Run XGrammar's compiled CPU mask on native Windows without Triton.

    The target model and generation algorithm remain unchanged. This copy can
    cost time, which is deliberately included in the measured generation time.
    """

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, input_ids, scores):
        device, dtype = scores.device, scores.dtype
        masked = self.inner(input_ids, scores.float().cpu())
        return masked.to(device=device, dtype=dtype)


class ConstrainedExtractorModel(ProfiledV2QwenJudge):
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        try:
            import xgrammar as xgr
        except ImportError as exc:
            raise RuntimeError("需要安装锁定的 xgrammar 依赖后才能运行 4.2A") from exc
        started = time.perf_counter()
        info = xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=self.model.config.vocab_size)
        compiler = xgr.GrammarCompiler(info)
        grammar = compiler.compile_json_schema(json.dumps(extractor_json_schema(), ensure_ascii=False))
        self._processor_factory = lambda: TorchCompatibleProcessor(xgr.contrib.hf.LogitsProcessor(grammar))
        self.grammar_compile_seconds = time.perf_counter() - started
        self.constraint_enabled = False

    def _generate(self, messages, max_new_tokens=768):
        if not self.constraint_enabled:
            return super()._generate(messages, max_new_tokens=max_new_tokens)
        with constrained_generation(self.model, self._processor_factory):
            return super()._generate(messages, max_new_tokens=max_new_tokens)

    def _generate_batch(self, message_batches, max_new_tokens=768):
        if not self.constraint_enabled:
            return super()._generate_batch(message_batches, max_new_tokens=max_new_tokens)
        if len(message_batches) != 1:
            raise ValueError("4.2A constrained Extractor only supports batch=1")
        with constrained_generation(self.model, self._processor_factory):
            return super()._generate_batch(message_batches, max_new_tokens=max_new_tokens)
