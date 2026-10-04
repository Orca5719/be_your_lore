"""Benchmark 4B continuous extractor command line interface."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from agent_pipeline_v2_2.runner import atomic_write_json
from agent_pipeline_v3.model import ProfiledV2QwenJudge

from .cli import ROOT, REPORT_ROOT, ensure_manifest, validate_inputs
from .continuous import QwenContinuousEngine, compare_continuous_single_story, run_continuous_cases
from .continuous_report import compare_case_outputs, render_continuous_table, static_wasted_steps, summarize_continuous
from .report import compare_extractions


CAPACITIES = (1, 2, 4, 8)
DEFAULT_STATIC = REPORT_ROOT / "benchmark_4A_20261004T050254Z_1774b9"
SCHEMA = "benchmark-4B-continuous-manifest-v1"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_manifest(identity: dict, paths: list[Path]) -> dict:
    return {"schema_version": SCHEMA, "identity": identity,
            "file_hashes": {str(path): _sha(path) for path in paths}}


def select_wall_seconds(result_wall: float | None, *, existing_count: int,
                        previous_wall: float | None) -> float | None:
    if existing_count == 0:
        return result_wall
    if existing_count == 24 and result_wall is None:
        return previous_wall
    return None


def _static_inputs(path: Path) -> tuple[dict[int, dict], list[Path]]:
    files = [path / "manifest.json"]
    if not files[0].is_file():
        raise ValueError(f"4A result directory missing manifest: {path}")
    static: dict[int, dict] = {}
    for capacity in CAPACITIES:
        directory = path / f"batch_{capacity}"
        summary = directory / "summary.json"
        calls = directory / "inference_calls.jsonl"
        rows = directory / "extractor_runs.jsonl"
        if not all(item.is_file() for item in (summary, calls, rows)):
            raise ValueError(f"4A source lacks batch {capacity} outputs")
        files.extend([summary, calls, rows])
        static[capacity] = _json(summary)
        static[capacity]["wasted_finished_steps"] = static_wasted_steps(_jsonl(calls))
        static[capacity]["rows"] = _jsonl(rows)
    return static, files


def _inputs(static_result: Path):
    source = validate_inputs()
    static, source_files = _static_inputs(static_result)
    files = source_files + [ROOT / name for name in (
        "agent_pipeline_v4/continuous.py", "agent_pipeline_v4/continuous_report.py",
        "agent_pipeline_v4/cli_continuous.py")]
    identity = {"baseline": source["identity"], "static_source": str(static_result.resolve()),
                "capacities": CAPACITIES,
                "generation": "Benchmark 3 greedy NF4 BF16; Qwen3 forward KV-cache continuous decode"}
    return source, static, make_manifest(identity, files)


def _summarize(directory: Path, capacity: int, inputs: dict, wall_seconds, static: dict):
    rows = _jsonl(directory / "extractor_runs.jsonl")
    steps = _jsonl(directory / "forward_steps.jsonl")
    requests = _jsonl(directory / "request_traces.jsonl")
    if len({step["step_id"] for step in steps}) != len(steps):
        raise ValueError("duplicate continuous forward step IDs")
    request_ids = {record["call_id"] for record in requests}
    if len(request_ids) != len(requests):
        raise ValueError("duplicate continuous request IDs")
    for row in rows:
        extraction = row["result"]["benchmark_stages"]["extraction"]
        for window in extraction.get("calls", []):
            for attempt in window.get("attempts", []):
                call_id = attempt.get("timing", {}).get("call_id")
                if call_id and call_id not in request_ids:
                    raise ValueError(f"attempt has no continuous request trace: {call_id}")
    quality = compare_extractions(inputs["cases"], inputs["source_rows"], rows, inputs["review"])
    performance = summarize_continuous(rows, steps, requests, capacity=capacity,
                                       wall_seconds=wall_seconds, scheduler=None)
    differences = compare_case_outputs(static[capacity]["rows"], rows)
    summary = {"schema_version": "benchmark-4B-continuous-summary-v1",
               "quality": quality, "performance": performance,
               "difference_case_ids": [item["case_id"] for item in differences],
               "decoded_output_comparison": "text_and_generated_token_count; raw_token_ids_unavailable_in_4A",
               "run_complete": len(rows) == 24,
               "extraction_complete": len(rows) == 24 and all(row["status"] == "ok" for row in rows),
               "static_reference": {"wall_seconds": static[capacity]["performance"]["wall_seconds"],
                                    "wasted_finished_steps": static[capacity]["wasted_finished_steps"]}}
    atomic_write_json(directory / "summary.json", summary)
    atomic_write_json(directory / "case_differences.json", differences)
    atomic_write_json(directory / "pending_event_review.json", quality["pending_events"])
    atomic_write_json(directory / "changed_case_ids.json", quality["changed_case_ids"])
    return summary


def _overview(output: Path, static: dict):
    summaries = [_json(output / f"capacity_{capacity}" / "summary.json") for capacity in CAPACITIES
                 if (output / f"capacity_{capacity}" / "summary.json").exists()]
    failures = [_json(output / f"capacity_{capacity}" / "failed.json") for capacity in CAPACITIES
                if (output / f"capacity_{capacity}" / "failed.json").exists()]
    if not summaries and not failures:
        return summaries
    atomic_write_json(output / "benchmark_4B_summary.json",
                      {"schema_version": "benchmark-4B-overview-v1", "capacities": summaries,
                       "failures": failures})
    (output / "benchmark_4B_summary.md").write_text(render_continuous_table(summaries, static, failures=failures), encoding="utf-8")
    with (output / "benchmark_4B_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        columns = ["capacity", "status_ok", "status_partial", "status_error", "requests", "model_steps",
                   "actual_avg_decode_batch", "input_tokens", "padded_input_tokens", "output_tokens",
                   "ttft_p50_ms", "ttft_p95_ms", "queue_wait_p50_ms", "queue_wait_p95_ms",
                   "prefill_seconds", "decode_seconds", "llm_seconds", "wall_seconds",
                   "non_model_overhead_seconds", "scheduler_seconds", "decode_tokens_per_second",
                   "slot_utilization", "finished_request_waste_steps", "peak_allocated", "peak_reserved",
                   "recall", "precision", "pending_events", "changed_cases", "static_wall_seconds",
                   "static_wasted_finished_steps", "difference_case_ids"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in summaries:
            p, q = item["performance"], item["quality"]
            writer.writerow({**{name: p.get(name) for name in columns if name in p},
                             "status_ok": p["status_counts"]["ok"],
                             "status_partial": p["status_counts"]["partial"],
                             "status_error": p["status_counts"]["error"],
                             "recall": q["candidate"]["recall"],
                             "precision": q["candidate"]["precision"],
                             "pending_events": q["pending_count"],
                             "changed_cases": ",".join(q["changed_case_ids"]),
                             "static_wall_seconds": item["static_reference"]["wall_seconds"],
                             "static_wasted_finished_steps": item["static_reference"]["wasted_finished_steps"],
                             "difference_case_ids": ",".join(item["difference_case_ids"])})
    return summaries


def run_command(args):
    inputs, static, manifest = _inputs(args.static_result.resolve())
    if args.output:
        output = args.output.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = REPORT_ROOT / f"benchmark_4B_{stamp}_{uuid.uuid4().hex[:6]}"
    output.mkdir(parents=True, exist_ok=True)
    ensure_manifest(output / "manifest.json", manifest)
    print("RESULT_DIR=" + str(output), flush=True)
    print("正在加载一次 Benchmark 3 模型……", flush=True)
    adapter = ProfiledV2QwenJudge("cuda")
    print("[SMOKE] 核对单请求 forward 解码与 Benchmark 3 输出", flush=True)
    compare_continuous_single_story(adapter, "雷捂住左胸，感到疼痛。")
    errors = 0
    for capacity in args.capacities:
        directory = output / f"capacity_{capacity}"
        directory.mkdir(parents=True, exist_ok=True)
        existing = _jsonl(directory / "extractor_runs.jsonl") if (directory / "extractor_runs.jsonl").exists() else []
        print(f"[CAPACITY {capacity}] {'续跑' if existing else '预热后运行'} {len(existing)}/24", flush=True)
        engine = QwenContinuousEngine(adapter)
        try:
            result = run_continuous_cases(inputs["cases"], engine, capacity, directory,
                                          warmup=not args.no_warmup and not existing,
                                          progress=lambda done, total, case_id: print(f"  [{capacity}] {done}/{total} {case_id}", flush=True))
            previous = _json(directory / "summary.json") if (directory / "summary.json").exists() else {}
            previous_wall = previous.get("performance", {}).get("wall_seconds")
            wall = select_wall_seconds(result["wall_seconds"], existing_count=len(existing),
                                       previous_wall=previous_wall)
            summary = _summarize(directory, capacity, inputs, wall, static)
            if not summary["extraction_complete"]:
                print(f"[CAPACITY {capacity}] 提取含partial/error，已保留原样供质量对照", flush=True)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            atomic_write_json(directory / "failed.json", {"capacity": capacity, "status": "oom", "error": str(exc)})
            print(f"[CAPACITY {capacity}] 显存不足，未降档：{exc}", flush=True)
            errors += 1
        adapter.clear_traces()
        _overview(output, static)
    print("SUMMARY=" + str(output / "benchmark_4B_summary.md"), flush=True)
    return 2 if errors else 0


def summary_command(args):
    inputs, static, manifest = _inputs(args.static_result.resolve())
    output = args.result_dir.resolve()
    ensure_manifest(output / "manifest.json", manifest)
    for capacity in CAPACITIES:
        directory = output / f"capacity_{capacity}"
        if (directory / "extractor_runs.jsonl").exists():
            previous = _json(directory / "summary.json") if (directory / "summary.json").exists() else {}
            wall = previous.get("performance", {}).get("wall_seconds")
            _summarize(directory, capacity, inputs, wall, static)
    summaries = _overview(output, static)
    if not summaries:
        raise ValueError("no continuous capacity data to summarize")
    print("SUMMARY=" + str(output / "benchmark_4B_summary.md"), flush=True)
    return 0


def validate_command(args):
    inputs, static, _ = _inputs(args.static_result.resolve())
    print(json.dumps({"status": "ok", "stories": len(inputs["cases"]),
                      "capacities": CAPACITIES,
                      "static_source": str(args.static_result.resolve()),
                      "static_wasted_steps": {str(k): v["wasted_finished_steps"] for k, v in static.items()}},
                     ensure_ascii=False, indent=2))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark 4B continuous extractor batching")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-continuous")
    validate.add_argument("--static-result", type=Path, default=DEFAULT_STATIC)
    validate.set_defaults(handler=validate_command)
    run = sub.add_parser("run-continuous")
    run.add_argument("--device", choices=("cuda",), default="cuda")
    run.add_argument("--capacities", nargs="+", type=int, choices=CAPACITIES, default=list(CAPACITIES))
    run.add_argument("--static-result", type=Path, default=DEFAULT_STATIC)
    run.add_argument("--output", type=Path)
    run.add_argument("--no-warmup", action="store_true", help="local debugging only")
    run.set_defaults(handler=run_command)
    summary = sub.add_parser("summary-continuous")
    summary.add_argument("--result-dir", type=Path, required=True)
    summary.add_argument("--static-result", type=Path, default=DEFAULT_STATIC)
    summary.set_defaults(handler=summary_command)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"错误：{exc}\n")
