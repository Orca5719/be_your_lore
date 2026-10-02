from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from agent_pipeline_v2_2.scoring import score_system
from agent_pipeline_v3.metrics import aggregate_profile
from agent_pipeline_v3.model import MODEL, REVISION, ProfiledV2QwenJudge

from .extractor_audit import run_extractor_audit
from .extractor_report import write_extractor_markdown
from .frozen import load_and_validate_source
from .lean_extractor import PROMPT_SHA256, extract_events_lean
from .lean_extractor_benchmark import run_extraction_cases, summarize_extractor, write_extractor_outputs
from .lean_judge import judge_frozen_retrieval
from .report import write_outputs
from .runner import run_rows, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "agent_pipeline_v3" / "reports" / "benchmark_3_20260927T142334Z_3ceb30"
DATASET = ROOT / "evaluation" / "story_benchmark_24_v2.json"
REVIEW = ROOT / "agent_pipeline_v2_2" / "reports" / "benchmark_2_2_20260926T122502Z_ecbff6" / "review.json"
REPORT_ROOT = ROOT / "agent_pipeline_v3_2" / "reports"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _output_path(value: Path | None) -> Path:
    if value:
        path = value.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = REPORT_ROOT / f"benchmark_3_2A_{stamp}_{uuid.uuid4().hex[:6]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ratio_change(new, old):
    return None if not old else (new - old) / old




def _candidate_review(review: dict) -> dict:
    sections = ("event_mapping", "finding_mapping", "system_event_labels", "system_finding_labels", "reasoning_support_labels")
    result = {section: {} for section in sections}
    for section in sections:
        for key, value in review.get(section, {}).items():
            if key.startswith("candidate:"):
                result[section][key[len("candidate:"):]] = value
    return result

def _comparisons(source_rows, lean_rows):
    old_by_case = {row["case_id"]: row for row in source_rows}
    values = []
    for row in lean_rows:
        case_id = row["case_id"]
        old_items = {item["event_id"]: item for item in old_by_case[case_id]["result"]["benchmark_stages"]["judge"]["items"]}
        for item in row["result"]["judge"]["items"]:
            old = old_items[item["event_id"]]
            values.append({
                "case_id": case_id, "event_id": item["event_id"],
                "old_verdict": old.get("verdict"), "new_verdict": item.get("verdict"),
                "changed": old.get("verdict") != item.get("verdict"),
                "evidence_ids": "|".join(item.get("evidence_ids", [])),
                "chunk_ids": "|".join(c.get("chunk_id", "") for c in item.get("citations", [])),
            })
    return values


def _summarize(directory: Path, source: dict, rows: list[dict], model_load_seconds: float) -> dict:
    calls = [call for row in rows for call in row.get("calls", [])]
    profile = aggregate_profile(calls, rows, model_load_seconds)
    dataset = _read(DATASET)
    review = _read(REVIEW)
    quality = score_system(dataset["cases"], rows, _candidate_review(review), 5)
    source_profile = _read(source["source"] / "profile_summary.json")
    baseline_judge = source_profile["components"]["judge"]
    reference = _read(source["source"] / "quality_reference.json")
    baseline_quality = reference["official_metrics"]
    lean_judge = profile["components"]["judge"]
    summary = {
        "schema_version": "agent-pipeline-v3.2-lean-judge-summary-v1",
        "source": str(source["source"]), "stories": len(rows), "facts": source["facts"],
        "model": MODEL, "revision": REVISION, "judge_batch_size": 8,
        "quality": quality, "baseline_quality": baseline_quality,
        "performance": profile, "baseline_judge": baseline_judge,
        "delta": {
            "output_token_change_ratio": _ratio_change(lean_judge["output_tokens"], baseline_judge["output_tokens"]),
            "prefill_change_ratio": _ratio_change(lean_judge["prefill_seconds"], baseline_judge["prefill_seconds"]),
            "decode_change_ratio": _ratio_change(lean_judge["decode_seconds"], baseline_judge["decode_seconds"]),
            "total_change_ratio": _ratio_change(lean_judge["total_seconds"], baseline_judge["total_seconds"]),
        },
    }
    comparisons = _comparisons(source["rows"], rows)
    write_outputs(directory, summary, comparisons, calls)
    return summary


def validate_command(args) -> int:
    source = load_and_validate_source(args.source_result)
    print(json.dumps({"status": "ok", "stories": source["stories"], "facts": source["facts"], "source_hashes": source["hashes"]}, ensure_ascii=False, indent=2))
    return 0


def run_command(args) -> int:
    if args.device != "cuda":
        raise ValueError("Benchmark 3.2A formal run is fixed to CUDA")
    if args.judge_batch_size != 8:
        raise ValueError("Benchmark 3.2A formal run is fixed to judge batch size 8")
    source = load_and_validate_source(args.source_result)
    directory = _output_path(args.output)
    manifest = {
        "schema_version": "agent-pipeline-v3.2-lean-judge-manifest-v1",
        "source": str(source["source"]), "source_hashes": source["hashes"],
        "model": MODEL, "revision": REVISION, "device": args.device, "judge_batch_size": 8,
        "prompt_sha256": __import__("hashlib").sha256((Path(__file__).parent / "prompts" / "lean_judge_v1.txt").read_bytes()).hexdigest(),
    }
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and _read(manifest_path) != manifest:
        raise ValueError("resume manifest mismatch")
    write_json_atomic(manifest_path, manifest)
    print("RESULT_DIR=" + str(directory), flush=True)
    existing_path = directory / "judge_runs.jsonl"
    existing_count = len(existing_path.read_text(encoding="utf-8").splitlines()) if existing_path.exists() else 0
    llm = ProfiledV2QwenJudge(args.device)
    if existing_count == 0:
        warmup = next(row for row in source["rows"] if row["result"]["benchmark_stages"]["retrieval"].get("items"))
        print("[WARMUP] " + warmup["case_id"], flush=True)
        llm.set_story_id("WARMUP")
        judge_frozen_retrieval(warmup["result"]["benchmark_stages"]["retrieval"], llm, 8)
        llm.clear_traces()
    def progress(value):
        if value.get("purpose") == "story_complete":
            print(f"[STORY {value['completed']}/24] {value['case_id']}", flush=True)
    rows = run_rows(source["rows"], llm, directory, 8, progress)
    metadata_path = directory / "run_metadata.json"
    metadata = _read(metadata_path) if metadata_path.exists() else {"model_load_seconds": 0.0}
    metadata["model_load_seconds"] = float(metadata.get("model_load_seconds", 0.0)) + float(llm.load_seconds)
    write_json_atomic(metadata_path, metadata)
    summary = _summarize(directory, source, rows, metadata["model_load_seconds"])
    print("SUMMARY=" + str(directory / "judge_lean_summary.md"), flush=True)
    print(json.dumps({"judge_accuracy": summary["quality"]["judge"]["accuracy"], "end_to_end_f1": summary["quality"]["end_to_end_conflict"]["f1"]}, ensure_ascii=False))
    return 0


def audit_extractor_command(args) -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or REPORT_ROOT / f"benchmark_3_2B_{stamp}_{uuid.uuid4().hex[:6]}").resolve()
    report = run_extractor_audit(args.source_result.resolve(), output)
    summary = write_extractor_markdown(output, report)
    print("AUDIT_DIR=" + str(output), flush=True)
    print("SUMMARY=" + str(summary), flush=True)
    print(json.dumps({
        "extractor_calls": report["extractor_calls"],
        "pure_waste_calls": report["cost_classes"]["pure_waste"]["calls"],
        "pure_waste_seconds": report["cost_classes"]["pure_waste"]["total_time"],
        "closure": report["closure"],
    }, ensure_ascii=False))
    return 0

def run_extractor_command(args) -> int:
    if args.device != "cuda":
        raise ValueError("Benchmark 3.2C formal run is fixed to CUDA")
    source = load_and_validate_source(args.source_result)
    dataset = _read(DATASET)
    previous_review = _read(REVIEW)
    source_profile = _read(source["source"] / "profile_summary.json")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or REPORT_ROOT / f"benchmark_3_2C_{stamp}_{uuid.uuid4().hex[:6]}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "agent-pipeline-v3.2-lean-extractor-manifest-v2",
        "source": str(source["source"]), "source_hashes": source["hashes"],
        "model": MODEL, "revision": REVISION, "device": args.device,
        "prompt_sha256": PROMPT_SHA256,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and _read(manifest_path) != manifest:
        raise ValueError("resume manifest mismatch")
    write_json_atomic(manifest_path, manifest)
    print("RESULT_DIR=" + str(output), flush=True)
    run_path = output / "extractor_runs.jsonl"
    existing = len(run_path.read_text(encoding="utf-8").splitlines()) if run_path.exists() else 0
    llm = ProfiledV2QwenJudge(args.device)
    if existing == 0:
        from agent_pipeline_v2_1.extractor import ExtractionRepairLLM
        warmup = dataset["cases"][0]
        print("[WARMUP] " + warmup["id"], flush=True)
        llm.set_story_id("WARMUP")
        extract_events_lean(warmup["story"], device=llm.device, llm=ExtractionRepairLLM(llm))
        llm.clear_traces()
    rows = run_extraction_cases(dataset["cases"], llm, output, lambda completed, total, case_id: print(f"[STORY {completed}/{total}] {case_id}", flush=True))
    metadata_path = output / "run_metadata.json"
    metadata = _read(metadata_path) if metadata_path.exists() else {"model_load_seconds": 0.0}
    metadata["model_load_seconds"] = float(metadata.get("model_load_seconds", 0.0)) + float(llm.load_seconds)
    write_json_atomic(metadata_path, metadata)
    summary = summarize_extractor(
        dataset=dataset, source_rows=source["rows"], lean_rows=rows, previous_review=previous_review,
        baseline_profile=source_profile, model_load_seconds=metadata["model_load_seconds"],
    )
    write_extractor_outputs(output, summary)
    print("SUMMARY=" + str(output / "extractor_lean_summary.md"), flush=True)
    print(json.dumps({"quality_gate": summary["quality_gate"]["status"], "pending_event_reviews": summary["review"]["pending_event_count"], "extractor_calls": summary["performance"]["lean"]["calls"], "output_tokens": summary["performance"]["lean"]["output_tokens"], "total_seconds": summary["performance"]["lean"]["total_seconds"]}, ensure_ascii=False))
    return 0

def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark 3.2 Lean Generation experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE)
    validate.set_defaults(func=validate_command)
    run = sub.add_parser("run-judge")
    run.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE)
    run.add_argument("--device", choices=("cuda",), default="cuda")
    run.add_argument("--judge-batch-size", type=int, default=8)
    run.add_argument("--output", type=Path)
    run.set_defaults(func=run_command)
    audit = sub.add_parser("audit-extractor")
    audit.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE)
    audit.add_argument("--output", type=Path)
    audit.set_defaults(func=audit_extractor_command)
    extract = sub.add_parser("run-extractor")
    extract.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE)
    extract.add_argument("--device", choices=("cuda",), default="cuda")
    extract.add_argument("--output", type=Path)
    extract.set_defaults(func=run_extractor_command)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print("错误：" + str(exc))
        return 2
