"""One clean-process worker for one Judge batch size and repeat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import judge_fixture_items
from .fixtures import validate_fixture


GIB = 2**30


def attach_runtime_metrics(
    result: dict,
    *,
    device: str,
    model_load_seconds,
    model_footprint_bytes,
    baseline_allocated_bytes,
    peak_allocated_bytes,
    peak_reserved_bytes,
) -> None:
    metrics = result.setdefault("metrics", {})
    metrics["model_load_seconds"] = model_load_seconds
    metrics["model_footprint_gib"] = model_footprint_bytes / GIB if model_footprint_bytes is not None else None
    metrics["model_cuda_allocated_after_load_gib"] = baseline_allocated_bytes / GIB if device == "cuda" and baseline_allocated_bytes is not None else None
    metrics["peak_allocated_gib"] = peak_allocated_bytes / GIB if device == "cuda" and peak_allocated_bytes is not None else None
    metrics["peak_reserved_gib"] = peak_reserved_bytes / GIB if device == "cuda" and peak_reserved_bytes is not None else None
    metrics["incremental_peak_allocated_gib"] = (peak_allocated_bytes - baseline_allocated_bytes) / GIB if device == "cuda" and peak_allocated_bytes is not None and baseline_allocated_bytes is not None else None


def run_worker(fixture_path: Path, output_path: Path, batch_size: int, device: str, repeat: int = 1) -> dict:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8-sig"))
    validate_fixture(fixture, expected_per_label=fixture["per_label"], expected_k=fixture["k"])
    from .model import MODEL, REVISION, V2QwenJudge

    llm = V2QwenJudge(device)
    torch = llm.torch
    actual_device = llm.device
    baseline = torch.cuda.memory_allocated() if actual_device == "cuda" else None
    warmup_items = fixture["items"][: min(batch_size, len(fixture["items"]))]
    judge_fixture_items(warmup_items, llm=llm, batch_size=batch_size)
    if actual_device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    result = judge_fixture_items(fixture["items"], llm=llm, batch_size=batch_size, progress=lambda step: print(f"  batch={batch_size} {step['completed']}/{step['total']} facts", flush=True))
    peak_allocated = torch.cuda.max_memory_allocated() if actual_device == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved() if actual_device == "cuda" else None
    result.update({
        "fixture_path": str(fixture_path),
        "fixture_items_digest": fixture["items_digest"],
        "repeat": repeat,
        "device": actual_device,
        "model": MODEL,
        "revision": REVISION,
        "precision": "NF4 4-bit weights, BF16 compute" if actual_device == "cuda" else "BF16",
        "warmup_fact_count": len(warmup_items),
    })
    attach_runtime_metrics(
        result,
        device=actual_device,
        model_load_seconds=llm.load_seconds,
        model_footprint_bytes=llm.model_footprint_bytes,
        baseline_allocated_bytes=baseline,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    try:
        result = run_worker(args.fixture, args.output, args.batch_size, args.device, args.repeat)
        return 0 if result["status"] == "ok" else 2
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        error = {
            "schema_version": "agent-pipeline-v2-judge-benchmark-result-v1",
            "batch_size": args.batch_size,
            "repeat": args.repeat,
            "status": "error",
            "oom": "out of memory" in str(exc).lower(),
            "error": str(exc),
            "error_type": type(exc).__name__,
            "metrics": {"facts_total": 0, "judge_accuracy": None},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(error, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("错误：" + str(exc), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
