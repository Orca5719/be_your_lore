"""Prompt-lookup speculative decoding for the frozen Extractor, batch size one."""

from __future__ import annotations

from contextlib import contextmanager
import time

from agent_pipeline_v3.model import ProfiledV2QwenJudge
from agent_pipeline_v3.profiling import validate_call_record


class CommitStreamer:
    """Observe committed token chunks without changing generation decisions."""

    def __init__(self, synchronize=None, clock=time.perf_counter):
        self.synchronize = synchronize
        self.clock = clock
        self.prompt_seen = False
        self.first_commit_at = None
        self.chunk_sizes: list[int] = []

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        if self.first_commit_at is None:
            if self.synchronize is not None:
                self.synchronize()
            self.first_commit_at = self.clock()
        self.chunk_sizes.append(int(value.numel()))

    def end(self):
        pass


@contextmanager
def prompt_lookup_generation(model, num_tokens: int, observed: dict, *, clock=time.perf_counter):
    """Use HF assisted generation only for this call, restoring generate afterward."""
    if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens < 1:
        raise ValueError("prompt lookup token count must be a positive integer")
    original = model.generate

    def generate(*args, **kwargs):
        if "prompt_lookup_num_tokens" in kwargs or "assistant_model" in kwargs:
            raise ValueError("another assisted generation mode is already active")
        inputs = kwargs.get("input_ids", args[0] if args else None)
        if inputs is None or inputs.shape[0] != 1:
            raise ValueError("Transformers 4.57.6 assisted generation requires batch size 1")
        # The frozen batch adapter's FirstTokenTimer is invoked by HF while
        # checking proposed candidates, before any token is committed. Remove
        # that measurement-only criterion and timestamp the first streamer
        # commit instead; EOS and max_new_tokens still come from generate().
        kwargs.pop("stopping_criteria", None)
        if kwargs.get("streamer") is not None:
            raise ValueError("unexpected streamer in frozen batch adapter")
        streamer = CommitStreamer(
            model_device_synchronize(model), clock=clock
        )
        kwargs["streamer"] = streamer
        kwargs["prompt_lookup_num_tokens"] = num_tokens
        observed["generate_start"] = clock()
        observed["streamer"] = streamer
        return original(*args, **kwargs)

    model.generate = generate
    try:
        yield
    finally:
        model.generate = original


def model_device_synchronize(model):
    # A fake model used in unit tests need not expose torch or CUDA.
    if getattr(model, "device", None) == "cuda" and hasattr(model, "torch"):
        return model.torch.cuda.synchronize
    return None


class PromptLookupExtractorModel(ProfiledV2QwenJudge):
    def __init__(self, device: str = "cuda", *, num_tokens: int = 5):
        super().__init__(device)
        self.num_tokens = num_tokens
        self.enabled = True

    def _generate_batch(self, message_batches, max_new_tokens=768):
        if not self.enabled:
            return super()._generate_batch(message_batches, max_new_tokens=max_new_tokens)
        if len(message_batches) != 1:
            raise ValueError("prompt lookup supports one Extractor request per generation")
        observed: dict = {}
        with prompt_lookup_generation(self.model, self.num_tokens, observed, clock=self.collector.clock):
            outputs = super()._generate_batch(message_batches, max_new_tokens=max_new_tokens)
        profile = self.last_batch_generation["profile"]
        streamer: CommitStreamer = observed["streamer"]
        if streamer.first_commit_at is not None:
            prefill = min(profile["total_time"], max(0.0, streamer.first_commit_at - observed["generate_start"]))
            profile["prefill_time"] = prefill
            profile["decode_time"] = profile["total_time"] - prefill
            profile["ttft_ms"] = prefill * 1000
        profile["commit_steps"] = len(streamer.chunk_sizes)
        profile["multi_token_steps"] = sum(size > 1 for size in streamer.chunk_sizes)
        profile["committed_tokens_from_streamer"] = sum(streamer.chunk_sizes)
        profile["extra_committed_tokens"] = sum(max(0, size - 1) for size in streamer.chunk_sizes)
        validate_call_record(profile)
        return outputs
