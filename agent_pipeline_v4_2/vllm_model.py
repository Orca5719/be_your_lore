"""4.2C: pinned prequantized model export and vLLM offline batch adapter."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import uuid

from qwen_judge import MODEL, REVISION

VLLM_VERSION = "0.18.0"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(directory: Path) -> dict:
    return {p.name: sha(p) for p in sorted(directory.iterdir())
            if p.is_file() and p.name != "worldcheck_export.json"}


def validate_artifact(directory: Path) -> dict:
    manifest = json.loads((directory / "worldcheck_export.json").read_text(encoding="utf-8"))
    if manifest.get("model") != MODEL or manifest.get("revision") != REVISION:
        raise ValueError("model artifact does not match the frozen Qwen revision")
    if manifest.get("files") != file_hashes(directory):
        raise ValueError("model artifact files changed or export is incomplete")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization_config", {})
    if not (quant.get("load_in_4bit") and quant.get("bnb_4bit_quant_type") == "nf4"
            and quant.get("bnb_4bit_use_double_quant") is True
            and quant.get("bnb_4bit_compute_dtype") == "bfloat16"):
        raise ValueError("artifact must preserve NF4, double quantization and BF16")
    if not list(directory.glob("*.safetensors")) or not (directory / "tokenizer_config.json").is_file():
        raise ValueError("artifact weights/tokenizer missing")
    return manifest


def export_model(directory: Path):
    """User-operated GPU load only; original model cache is never modified."""
    from agent_pipeline_v2_2.runner import atomic_write_json
    if directory.exists():
        validate_artifact(directory)
        print("MODEL_DIR=" + str(directory), flush=True)
        return
    from qwen_judge import QwenJudge
    staging = directory.with_name(directory.name + ".export-" + uuid.uuid4().hex[:8])
    staging.mkdir(parents=True)
    model = QwenJudge("cuda")
    model.model.save_pretrained(staging, safe_serialization=True)
    model.tokenizer.save_pretrained(staging)
    atomic_write_json(staging / "worldcheck_export.json", {
        "model": MODEL, "revision": REVISION, "files": file_hashes(staging),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "bitsandbytes")},
    })
    validate_artifact(staging)
    staging.rename(directory)
    print("MODEL_DIR=" + str(directory), flush=True)


def preflight() -> dict:
    if platform.system() != "Linux":
        raise RuntimeError("4.2C vLLM runs in WSL2/Linux; use export-model on Windows, then run in WSL2")
    versions = {p: importlib.metadata.version(p) for p in ("vllm", "torch", "transformers", "bitsandbytes")}
    if versions["vllm"] != VLLM_VERSION:
        raise RuntimeError(f"Use the isolated 4.2C environment with vllm=={VLLM_VERSION}")
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA GPU with BF16 support is required")
    return {"versions": versions, "python": platform.python_version(), "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
            "capability": list(torch.cuda.get_device_capability(0))}


def engine_options(model_dir: Path, batch_size: int, ngram: int, memory: float) -> dict:
    if batch_size not in (1, 2, 4) or ngram not in (0, 3, 5) or not 0 < memory < 1:
        raise ValueError("invalid batch/ngram/GPU memory configuration")
    options = dict(model=str(model_dir.resolve()), tokenizer=str(model_dir.resolve()),
                   dtype="bfloat16", quantization="bitsandbytes", load_format="bitsandbytes",
                   max_model_len=4096, max_num_seqs=batch_size, gpu_memory_utilization=memory,
                   tensor_parallel_size=1, seed=0, enable_prefix_caching=False,
                   enable_chunked_prefill=False, async_scheduling=False,
                   enforce_eager=True, generation_config="vllm",
                   disable_log_stats=False)
    if ngram:
        options["speculative_config"] = dict(method="ngram", num_speculative_tokens=ngram,
                                              prompt_lookup_min=2, prompt_lookup_max=5)
    return options


class VllmExtractorModel:
    device = "cuda"

    def __init__(self, model_dir: Path, batch_size: int, ngram: int, memory: float):
        preflight()
        validate_artifact(model_dir)
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        generation = json.loads((model_dir / "generation_config.json").read_text(encoding="utf-8"))
        eos = generation["eos_token_id"]
        self.eos_ids = [eos] if isinstance(eos, int) else eos
        self.engine = LLM(**engine_options(model_dir, batch_size, ngram, memory))
        self.sampling_factory = SamplingParams
        self.batch_size = batch_size
        self.sequence = 0
        self.load_seconds = time.perf_counter() - started
        self.last_batch_generation = {}

    def _generate_batch(self, message_batches, max_new_tokens=768):
        self.last_batch_generation = {}
        if not 0 < len(message_batches) <= self.batch_size:
            raise ValueError("request count exceeds configured batch limit")
        prompts = [{"prompt_token_ids": self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True)} for messages in message_batches]
        counts = [len(p["prompt_token_ids"]) for p in prompts]
        if max_new_tokens < 1 or any(n + max_new_tokens > 4096 for n in counts):
            raise ValueError("input plus output exceeds 4096 tokens; no truncation")
        params = self.sampling_factory(temperature=0, max_tokens=max_new_tokens, seed=0,
                                       repetition_penalty=1.0, skip_special_tokens=True,
                                       stop_token_ids=self.eos_ids)
        started = time.perf_counter()
        outputs = self.engine.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - started
        if len(outputs) != len(prompts):
            raise RuntimeError("vLLM returned wrong request count")
        # LLM.generate documents outputs in input order. Verify the prompt mapping too.
        for prompt, result in zip(prompts, outputs):
            if list(result.prompt_token_ids) != prompt["prompt_token_ids"] or len(result.outputs) != 1:
                raise RuntimeError("vLLM request mapping mismatch")
            if result.outputs[0].finish_reason not in ("stop", "length"):
                raise RuntimeError("vLLM request aborted or did not finish")
        generated = [len(r.outputs[0].token_ids) for r in outputs]
        self.sequence += 1
        call_id = "vllm-" + uuid.uuid4().hex
        profile = {"call_id": call_id, "batch_size": len(prompts), "input_tokens": sum(counts),
                   "output_tokens": sum(generated), "total_time": elapsed, "ttft_ms": None,
                   "prefill_time": None, "decode_time": None, "peak_allocated": None,
                   "peak_reserved": None, "backend": "vllm", "padded_input_tokens": 0}
        self.last_batch_generation = {"call_id": call_id, "profile": profile, "seconds": elapsed,
                                      "input_tokens": counts, "generated_tokens": generated,
                                      "truncated_indices": [i for i, r in enumerate(outputs)
                                                            if r.outputs[0].finish_reason == "length"]}
        return [r.outputs[0].text for r in outputs]
