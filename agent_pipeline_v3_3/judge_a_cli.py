"""Run two compact Judge variants against Benchmark 3 frozen facts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from agent_pipeline_v2_2.scoring import score_system
from agent_pipeline_v3.metrics import aggregate_profile
from agent_pipeline_v3.model import ProfiledV2QwenJudge
from agent_pipeline_v3_2.cli import _candidate_review
from agent_pipeline_v3_2.frozen import load_and_validate_source, sha256
from agent_pipeline_v3_2.runner import write_json_atomic

from .judge_a import judge_frozen_retrieval
from .judge_a_report import classification_metrics, verdict_changes, write_report
from .judge_a_runner import build_manifest, run_variant


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "agent_pipeline_v3" / "reports" / "benchmark_3_20260927T142334Z_3ceb30"
LEAN_RESULT = ROOT / "agent_pipeline_v3_2" / "reports" / "benchmark_3_2A_20261001T151227Z_cf125b" / "judge_lean_summary.json"
DATASET = ROOT / "evaluation" / "story_benchmark_24_v2.json"
REVIEW = ROOT / "agent_pipeline_v2_2" / "reports" / "benchmark_2_2_20260926T122502Z_ecbff6" / "review.json"
REPORT_ROOT = ROOT / "agent_pipeline_v3_3" / "reports"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _row(quality: dict, performance: dict) -> dict:
    return {"quality": quality, "classification": classification_metrics(quality["judge"]["confusion"]),
            "performance": performance}


def summarize(directory: Path, source: dict) -> dict:
    data = _read(DATASET)
    review = _candidate_review(_read(REVIEW))
    original = _read(source["source"] / "profile_summary.json")
    lean = _read(LEAN_RESULT)
    baseline_quality = _read(source["source"] / "quality_reference.json")["official_metrics"]
    variants = {}
    changes = []
    for variant in ("structured", "short-reason"):
        path = directory / variant / "judge_runs.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 24:
            raise ValueError(f"{variant}: 只完成{len(rows)}/24篇，不能生成正式对照")
        calls = [call for row in rows for call in row.get("calls", [])]
        profile = aggregate_profile(calls, rows, 0.0)
        quality = score_system(data["cases"], rows, review, 5)
        variants[variant] = {**_row(quality, profile["components"]["judge"]), "calls": calls,
                             "complete_stories": sum(row["status"] == "ok" for row in rows)}
        changes.extend(verdict_changes(source["rows"], rows, variant))
    summary = {
        "schema_version": "agent-pipeline-v3.3a-summary-v1",
        "source": str(source["source"]), "stories": 24, "facts": 80,
        "baseline": _row(baseline_quality, original["components"]["judge"]),
        "lean": _row(lean["quality"], lean["performance"]["components"]["judge"]),
        "variants": variants,
        "notice": "冻结Extraction与Retrieval；两个候选仅重新运行Judge。性能为单轮实测，质量来自固定开发集。",
    }
    write_report(directory, summary, changes)
    return summary


def run_command(args) -> int:
    if args.device != "cuda":
        raise ValueError("3.3A正式实验固定CUDA")
    source = load_and_validate_source(args.source_result)
    if args.output:
        directory = args.output.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = REPORT_ROOT / f"benchmark_3_3A_{stamp}_{uuid.uuid4().hex[:6]}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(source["hashes"], args.device)
    manifest["source"] = str(source["source"])
    manifest["implementation_hashes"]["judge_a_cli.py"] = sha256(Path(__file__))
    manifest["inputs"] = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                          for path in (DATASET, REVIEW, LEAN_RESULT)}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and _read(manifest_path) != manifest:
        raise ValueError("续跑manifest不匹配；不得混用旧结果")
    write_json_atomic(manifest_path, manifest)
    print("RESULT_DIR=" + str(directory), flush=True)
    remaining = []
    for variant in ("structured", "short-reason"):
        path = directory / variant / "judge_runs.jsonl"
        count = len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0
        if count < 24:
            remaining.append(variant)
    if not remaining:
        summary = summarize(directory, source)
        print("SUMMARY=" + str(directory / "judge_a_summary.md"), flush=True)
        return 0 if all(value["complete_stories"] == 24 for value in summary["variants"].values()) else 2
    llm = ProfiledV2QwenJudge(args.device)
    metadata_path = directory / "run_metadata.json"
    metadata = _read(metadata_path) if metadata_path.exists() else {"model_load_sessions": []}
    metadata["model_load_sessions"].append(float(llm.load_seconds))
    write_json_atomic(metadata_path, metadata)
    for variant in remaining:
        warmup = next(row for row in source["rows"] if row["result"]["benchmark_stages"]["retrieval"].get("items"))
        print(f"[WARMUP {variant}] {warmup['case_id']}", flush=True)
        llm.set_story_id("WARMUP")
        judge_frozen_retrieval(warmup["result"]["benchmark_stages"]["retrieval"], llm, variant, 8)
        llm.clear_traces()
        print(f"[VARIANT {variant}]", flush=True)
        run_variant(source["rows"], llm, directory / variant, variant,
                    lambda name, case_id, completed, total: print(f"  [{name} {completed}/{total}] {case_id}", flush=True))
    summary = summarize(directory, source)
    print("SUMMARY=" + str(directory / "judge_a_summary.md"), flush=True)
    return 0 if all(value["complete_stories"] == 24 for value in summary["variants"].values()) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark 3.3A compact Judge experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE)
    run_parser.add_argument("--device", choices=("cuda",), default="cuda")
    run_parser.add_argument("--output", type=Path)
    run_parser.set_defaults(handler=run_command)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, OSError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print("错误：" + str(exc), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
