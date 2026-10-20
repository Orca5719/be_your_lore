"""CLI for the standalone 3.3B Extractor coverage experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from agent_pipeline_v3_2.frozen import load_and_validate_source, read_jsonl
from agent_pipeline_v3_2.runner import write_json_atomic

from .extractor_b import PROMPT_PATH, extract_events_b
from .extractor_b_benchmark import render_report, run_cases, summarize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "agent_pipeline_v3_3" / "reports" / "benchmark_3_3_base_20261003T150503Z_e38d1b"
DATASET = ROOT / "evaluation" / "story_benchmark_24_v2.json"
REVIEW = ROOT / "agent_pipeline_v2_2" / "reports" / "benchmark_2_2_20260926T122502Z_ecbff6" / "review.json"
REPORT_ROOT = ROOT / "agent_pipeline_v3_3" / "reports"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest(source, device):
    from agent_pipeline_v3.model import MODEL, REVISION
    inputs = [
        DATASET, REVIEW, PROMPT_PATH,
        ROOT / "agent_pipeline_v2" / "extractor.py",
        ROOT / "agent_pipeline_v2_1" / "extractor.py",
        ROOT / "agent_pipeline_v3_3" / "coverage.py",
        ROOT / "agent_pipeline_v3_3" / "extractor_b.py",
        ROOT / "agent_pipeline_v3_3" / "extractor_b_benchmark.py",
        Path(__file__),
    ]
    return {
        "schema_version": "agent-pipeline-v3.3B-manifest-v1",
        "source": str(source["source"]), "source_hashes": source["hashes"],
        "input_hashes": {str(path.relative_to(ROOT)): _hash(path) for path in inputs},
        "model": MODEL, "revision": REVISION, "device": device,
        "benchmark_scope": "extractor-only-24-stories",
    }


def _ensure_manifest(directory, manifest):
    path = directory / "manifest.json"
    if path.exists() and _read(path) != manifest:
        raise ValueError("3.3B manifest不匹配；资料、模型、提示或代码已改变，请使用新结果目录")
    write_json_atomic(path, manifest)


def record_model_load(path: Path, seconds: float) -> None:
    metadata = _read(path) if path.exists() else {"model_load_seconds": 0.0}
    metadata["model_load_seconds"] += float(seconds)
    write_json_atomic(path, metadata)


def _summarize(directory, source):
    rows = read_jsonl(directory / "extractor_runs.jsonl")
    dataset = _read(DATASET)
    metadata = _read(directory / "run_metadata.json")
    report = summarize(dataset, source["rows"], _read(source["source"] / "profile_summary.json"),
                       rows, _read(REVIEW), metadata["model_load_seconds"])
    write_json_atomic(directory / "extractor_b_summary.json", report)
    (directory / "extractor_b_summary.md").write_text(render_report(report), encoding="utf-8")
    write_json_atomic(directory / "event_review.json", report["review"])
    print("SUMMARY=" + str(directory / "extractor_b_summary.md"), flush=True)
    return report


def run_command(args):
    if args.device != "cuda":
        raise ValueError("3.3B正式性能测试固定使用CUDA")
    source = load_and_validate_source(args.source_result)
    dataset = _read(DATASET)
    if len(dataset["cases"]) != 24 or {case["id"] for case in dataset["cases"]} != {row["case_id"] for row in source["rows"]}:
        raise ValueError("3.3B数据集必须与基座24篇故事一致")
    if args.output:
        directory = args.output.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = REPORT_ROOT / f"benchmark_3_3B_{stamp}_{uuid.uuid4().hex[:6]}"
    directory.mkdir(parents=True, exist_ok=True)
    _ensure_manifest(directory, _manifest(source, args.device))
    print("RESULT_DIR=" + str(directory), flush=True)
    existing = read_jsonl(directory / "extractor_runs.jsonl") if (directory / "extractor_runs.jsonl").exists() else []
    if len(existing) < 24:
        from agent_pipeline_v2_1.extractor import ExtractionRepairLLM
        from agent_pipeline_v3.model import ProfiledV2QwenJudge
        llm = ProfiledV2QwenJudge(args.device)
        record_model_load(directory / "run_metadata.json", llm.load_seconds)
        if not existing:
            case = dataset["cases"][0]
            print("[WARMUP] " + case["id"], flush=True)
            llm.set_story_id("WARMUP")
            extract_events_b(case["story"], device=llm.device, llm=ExtractionRepairLLM(llm))
            llm.clear_traces()
        rows = run_cases(dataset["cases"], llm, directory,
                         lambda done, total, case_id: print(f"[STORY {done}/{total}] {case_id}", flush=True))
    else:
        rows = existing
    report = _summarize(directory, source)
    print(json.dumps({"stories": report["stories"], "events": report["events"],
                      "uncovered_spans": report["attempts"]["uncovered_span_count"],
                      "pending_event_reviews": report["review"]["pending_event_count"],
                      "quality_gate": report["quality_gate"]}, ensure_ascii=False), flush=True)
    return 0 if len(rows) == 24 and all(row["status"] == "ok" for row in rows) else 2


def summary_command(args):
    directory = args.result_dir.resolve()
    manifest = _read(directory / "manifest.json")
    source = load_and_validate_source(Path(manifest["source"]))
    if manifest != _manifest(source, manifest["device"]):
        raise ValueError("3.3B结果与当前输入或代码不匹配，拒绝混合重算")
    _summarize(directory, source)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark 3.3B deterministic coverage")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE)
    run.add_argument("--device", choices=("cuda",), default="cuda")
    run.add_argument("--output", type=Path)
    run.set_defaults(handler=run_command)
    summary = sub.add_parser("summary")
    summary.add_argument("--result-dir", type=Path, required=True)
    summary.set_defaults(handler=summary_command)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print("错误：" + str(exc), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
