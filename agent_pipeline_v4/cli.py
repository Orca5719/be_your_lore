"""Benchmark 4A command line interface."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from agent_pipeline_v2_2.runner import atomic_write_json
from agent_pipeline_v2_2.schema import stable_digest
from agent_pipeline_v3.model import MODEL, REVISION, ProfiledV2QwenJudge

from .report import compare_extractions, render_table, summarize_static
from .static import compare_single_story, run_static_cases


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation/story_benchmark_24_v2.json"
SOURCE = ROOT / "agent_pipeline_v3/reports/benchmark_3_20260927T142334Z_3ceb30"
REVIEW = ROOT / "agent_pipeline_v2_2/reports/benchmark_2_2_20260926T122502Z_ecbff6/review.json"
REPORT_ROOT = ROOT / "agent_pipeline_v4/reports"
SIZES = (1, 2, 4, 8)
MANIFEST_SCHEMA = "benchmark-4A-static-manifest-v1"
FROZEN_B3_FILES = {
    "agent_pipeline_v2/extractor.py": "67d852e99e8af17cf68a03e79cfd1374cc50c45690ff2ba7a6775b114aa55b64",
    "agent_pipeline_v2_1/extractor.py": "c63c0f4539a9404c062c0c8405304dffa49b884c2663c14e83c5f9bd7a8d7699",
    "agent_pipeline_v2/model.py": "242a1680af9367fd37e888ebb717ebfa67d8964e3d00c3a08917aa5305d5aa6f",
    "agent_pipeline_v2/batch_llm.py": "8555f8088fd5624bb1ee631fe2c63af974fdbf77a17547b79ad72c5b4fdfe34e",
    "qwen_judge.py": "a0ed992e92dd29db70a6d357251837b8439b30209dc05af091d9e5ba43765ee8",
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen_files(expected: dict[Path, str]) -> None:
    for path, digest in expected.items():
        content = path.read_bytes().replace(bytes((13, 10)), bytes((10,)))
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"frozen Benchmark 3 file changed: {path}")


def validate_inputs() -> dict:
    required = [
        DATASET, REVIEW, SOURCE / "story_runs.jsonl", SOURCE / "manifest.json", SOURCE / "profile_summary.json",
        ROOT / "agent_pipeline_v2/extractor.py",
        ROOT / "agent_pipeline_v2/prompts/extractor_v1.txt",
        ROOT / "agent_pipeline_v2_1/extractor.py",
        ROOT / "agent_pipeline_v2/model.py",
        ROOT / "agent_pipeline_v2/batch_llm.py",
        ROOT / "qwen_judge.py",
        ROOT / "agent_pipeline_v3/model.py",
        ROOT / "agent_pipeline_v4/static.py",
        ROOT / "agent_pipeline_v4/report.py",
        ROOT / "agent_pipeline_v4/cli.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Benchmark 4A missing input files: " + ", ".join(missing))
    dataset = _json(DATASET)
    cases = dataset.get("cases", [])
    source_rows = _jsonl(SOURCE / "story_runs.jsonl")
    ids = [case.get("id") for case in cases]
    if len(cases) != 24 or len(source_rows) != 24 or len(ids) != len(set(ids)):
        raise ValueError("Benchmark 4A requires the frozen 24-story dataset and 24 source rows")
    if {row.get("case_id") for row in source_rows} != set(ids):
        raise ValueError("Benchmark 3 source story IDs do not match dataset")
    reference = _json(SOURCE / "manifest.json")
    identity = reference.get("identity", {})
    if identity.get("dataset") != stable_digest(dataset):
        raise ValueError("Benchmark 3 source manifest is invalid")
    if identity.get("inputs", {}).get("model_id") != MODEL or identity.get("inputs", {}).get("model_revision") != REVISION:
        raise ValueError("Benchmark 3 model identity changed")
    frozen = {ROOT / path: digest for path, digest in FROZEN_B3_FILES.items()}
    for name in ("agent_pipeline_v2/prompts/extractor_v1.txt", "agent_pipeline_v3/model.py"):
        frozen[ROOT / name] = identity["inputs"][name]
    check_frozen_files(frozen)
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): _sha(path) for path in required}
    return {"cases": cases, "source_rows": source_rows, "review": _json(REVIEW),
            "identity": {"files": hashes, "model": MODEL, "revision": REVISION, "device": "cuda",
                         "sizes": SIZES, "generation": "Benchmark-3 greedy NF4 BF16 max_new_tokens=1536"}}


def ensure_manifest(path: Path, manifest: dict) -> None:
    if path.exists():
        if _json(path) != manifest:
            raise ValueError("resume manifest mismatch; create a new result directory")
        return
    atomic_write_json(path, manifest)


def _directory(output: Path | None) -> Path:
    if output:
        return output.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPORT_ROOT / f"benchmark_4A_{stamp}_{uuid.uuid4().hex[:6]}"


def _quality_and_performance(directory: Path, batch_size: int, inputs: dict, wall_seconds: float) -> dict:
    rows = _jsonl(directory / "extractor_runs.jsonl")
    calls = _jsonl(directory / "inference_calls.jsonl")
    call_ids = {call["call_id"] for call in calls}
    if len(call_ids) != len(calls):
        raise ValueError("duplicate LLM call IDs in saved traces")
    for row in rows:
        extraction = row["result"]["benchmark_stages"]["extraction"]
        for window in extraction.get("calls", []):
            for attempt in window.get("attempts", []):
                call_id = attempt.get("timing", {}).get("call_id")
                if call_id and call_id not in call_ids:
                    raise ValueError(f"attempt has no LLM trace: {call_id}")
    quality = compare_extractions(inputs["cases"], inputs["source_rows"], rows, inputs["review"])
    performance = summarize_static(rows, calls, wall_seconds=wall_seconds, batch_size=batch_size)
    summary = {"schema_version": "benchmark-4A-static-summary-v1", "quality": quality, "performance": performance,
               "complete": len(rows) == 24 and all(row["status"] == "ok" for row in rows)}
    atomic_write_json(directory / "summary.json", summary)
    atomic_write_json(directory / "pending_event_review.json", quality["pending_events"])
    atomic_write_json(directory / "changed_case_ids.json", quality["changed_case_ids"])
    return summary


def _write_overview(output: Path, inputs: dict) -> list[dict]:
    summaries = []
    failures = []
    for size in SIZES:
        path = output / f"batch_{size}" / "summary.json"
        if path.exists():
            summaries.append(_json(path))
        failure = output / f"batch_{size}" / "failed.json"
        if failure.exists():
            failures.append(_json(failure))
    if not summaries and not failures:
        return summaries
    baseline = summaries[0]["quality"]["baseline"] if summaries else compare_extractions(inputs["cases"], inputs["source_rows"], [], inputs["review"])["baseline"]
    atomic_write_json(output / "benchmark_4A_summary.json", {"schema_version": "benchmark-4A-overview-v1", "sizes": summaries, "failures": failures})
    source_profile = _json(SOURCE / "profile_summary.json")["components"]["extractor"]
    (output / "benchmark_4A_summary.md").write_text(render_table(summaries, baseline, failures=failures,
                                                                  baseline_performance=source_profile), encoding="utf-8")
    with (output / "benchmark_4A_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["batch_size", "status", "actual_avg_batch", "stories", "events", "calls", "retry_requests", "coverage_recovery_requests", "input_tokens", "padded_input_tokens", "output_tokens", "ttft_p50_ms", "ttft_p95_ms", "prefill_seconds", "decode_seconds", "llm_seconds", "wall_seconds", "decode_tokens_per_second", "stories_per_second", "peak_allocated", "peak_reserved", "recall", "precision", "pending_events", "complete"])
        writer.writeheader()
        for item in summaries:
            p, q = item["performance"], item["quality"]
            writer.writerow({"batch_size": p["batch_size"], "status": "ok" if item["complete"] else "partial", "actual_avg_batch": p["actual_avg_batch"], "stories": p["stories"], "events": p["events"], "calls": p["llm_calls"], "retry_requests": p["retry_requests"], "coverage_recovery_requests": p["coverage_recovery_requests"], "input_tokens": p["input_tokens"], "padded_input_tokens": p["padded_input_tokens"], "output_tokens": p["output_tokens"], "ttft_p50_ms": p["ttft_p50_ms"], "ttft_p95_ms": p["ttft_p95_ms"], "prefill_seconds": p["prefill_seconds"], "decode_seconds": p["decode_seconds"], "llm_seconds": p["llm_seconds"], "wall_seconds": p["wall_seconds"], "decode_tokens_per_second": p["decode_tokens_per_second"], "stories_per_second": p["stories_per_second"], "peak_allocated": p["peak_allocated"], "peak_reserved": p["peak_reserved"], "recall": q["candidate"]["recall"], "precision": q["candidate"]["precision"], "pending_events": q["pending_count"], "complete": item["complete"]})
        for failure in failures:
            writer.writerow({"batch_size": failure["batch_size"], "status": failure["status"], "complete": False})
    return summaries


def run_command(args) -> int:
    inputs = validate_inputs()
    output = _directory(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ensure_manifest(output / "manifest.json", {"schema_version": MANIFEST_SCHEMA, "identity": inputs["identity"]})
    print("RESULT_DIR=" + str(output), flush=True)
    print("正在加载一次 Benchmark 3 模型……", flush=True)
    model = ProfiledV2QwenJudge("cuda")
    print("[SMOKE] 比较原单条生成与 batch=1 的同篇提取结果", flush=True)
    compare_single_story(model, inputs["cases"][0]["story"])
    errors = 0
    for size in args.batch_sizes:
        directory = output / f"batch_{size}"
        directory.mkdir(parents=True, exist_ok=True)
        existing = _jsonl(directory / "extractor_runs.jsonl") if (directory / "extractor_runs.jsonl").exists() else []
        completed_before = sum(row.get("status") == "ok" for row in existing)
        if existing and completed_before < 24:
            print(f"[BATCH {size}] 续跑 {completed_before}/24；本档位耗时不用于正式横向对比", flush=True)
        else:
            print(f"[BATCH {size}] 预热一篇，然后处理24篇", flush=True)
        model.set_story_id(None)
        try:
            result = run_static_cases(inputs["cases"], model, size, directory,
                                      warmup=not args.no_warmup,
                                      progress=lambda done, total, case_id: print(f"  [{size}] {done}/{total} {case_id}", flush=True))
            wall_seconds = result["wall_seconds"]
            if existing and completed_before < 24:
                wall_seconds = None
            elif wall_seconds is None and (directory / "summary.json").exists():
                wall_seconds = _json(directory / "summary.json")["performance"]["wall_seconds"]
            summary = _quality_and_performance(directory, size, inputs, wall_seconds)
            if summary["complete"]:
                (directory / "failed.json").unlink(missing_ok=True)
            if not summary["complete"]:
                errors += 1
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            atomic_write_json(directory / "failed.json", {"batch_size": size, "status": "oom", "error": str(exc)})
            print(f"[BATCH {size}] 显存不足；未降档：{exc}", flush=True)
            errors += 1
        model.clear_traces()
        _write_overview(output, inputs)
    print("SUMMARY=" + str(output / "benchmark_4A_summary.md"), flush=True)
    return 2 if errors else 0


def summary_command(args) -> int:
    inputs = validate_inputs()
    output = args.result_dir.resolve()
    ensure_manifest(output / "manifest.json", {"schema_version": MANIFEST_SCHEMA, "identity": inputs["identity"]})
    for size in SIZES:
        directory = output / f"batch_{size}"
        if (directory / "extractor_runs.jsonl").exists():
            previous = _json(directory / "summary.json") if (directory / "summary.json").exists() else {}
            wall_seconds = previous.get("performance", {}).get("wall_seconds")
            _quality_and_performance(directory, size, inputs, wall_seconds)
    summaries = _write_overview(output, inputs)
    if not summaries:
        raise ValueError("no completed batch data to summarize")
    print("SUMMARY=" + str(output / "benchmark_4A_summary.md"), flush=True)
    return 0


def validate_command(_args) -> int:
    inputs = validate_inputs()
    print(json.dumps({"status": "ok", "stories": len(inputs["cases"]), "model": MODEL,
                      "revision": REVISION, "batch_sizes": SIZES}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark 4A frozen Extractor static batching")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.set_defaults(handler=validate_command)
    run = sub.add_parser("run-static")
    run.add_argument("--device", choices=("cuda",), default="cuda")
    run.add_argument("--batch-sizes", nargs="+", type=int, choices=SIZES, default=list(SIZES))
    run.add_argument("--output", type=Path)
    run.add_argument("--no-warmup", action="store_true", help="for local debugging only")
    run.set_defaults(handler=run_command)
    summary = sub.add_parser("summary")
    summary.add_argument("--result-dir", type=Path, required=True)
    summary.set_defaults(handler=summary_command)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
