"""Clean-process worker for one Benchmark 2.1A prompt and repeat."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from agent_pipeline_v2.benchmark_worker import attach_runtime_metrics

from .oracle_benchmark import run_oracle_items


def run_worker(fixture_path: Path, output_path: Path, prompt_version: str, batch_size: int, device: str, repeat: int) -> dict:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8-sig"))
    items = fixture.get("items", [])
    if fixture.get("status") != "reviewed" or len(items) != 72:
        raise ValueError("Benchmark 2.1A要求reviewed状态的72条Oracle fixture")
    from agent_pipeline_v2.model import MODEL, REVISION, V2QwenJudge

    llm = V2QwenJudge(device)
    torch = llm.torch
    actual_device = llm.device
    baseline = torch.cuda.memory_allocated() if actual_device == "cuda" else None
    warmup_items = items[: min(batch_size, len(items))]
    run_oracle_items(warmup_items, llm, prompt_version, batch_size)
    if actual_device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    result = run_oracle_items(
        items, llm, prompt_version, batch_size,
        progress=lambda step: print(f"  {prompt_version} {step['completed']}/{step['total']} facts", flush=True),
    )
    peak_allocated = torch.cuda.max_memory_allocated() if actual_device == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved() if actual_device == "cuda" else None
    result.update({
        "fixture_path": str(fixture_path),
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "repeat": repeat,
        "device": actual_device,
        "model": MODEL,
        "revision": REVISION,
        "precision": "NF4 4-bit weights, BF16 compute" if actual_device == "cuda" else "BF16",
        "warmup_fact_count": len(warmup_items),
    })
    attach_runtime_metrics(
        result, device=actual_device, model_load_seconds=llm.load_seconds,
        model_footprint_bytes=llm.model_footprint_bytes, baseline_allocated_bytes=baseline,
        peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-version", choices=("v1", "v2.1"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    try:
        result = run_worker(args.fixture, args.output, args.prompt_version, args.batch_size, args.device, args.repeat)
        # partial是有效测量结果：解析失败率本身就是benchmark指标。
        return 2 if result["status"] == "error" else 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        error = {
            "schema_version": "agent-pipeline-v2.1-oracle-run-v1",
            "prompt_version": args.prompt_version,
            "batch_size": args.batch_size,
            "repeat": args.repeat,
            "status": "error",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "metrics": {"facts_total": 0, "accuracy": None},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(error, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("错误：" + str(exc), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
