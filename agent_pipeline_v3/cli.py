from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

from agent_pipeline_v2_2.runner import atomic_write_json
from agent_pipeline_v2_2.story_pipeline import process_story

from .metrics import aggregate_profile
from .model import MODEL, REVISION, ProfiledV2QwenJudge
from .quality import compare_quality
from .report import write_profile_outputs
from .runner import build_manifest, ensure_manifest, load_story_rows, run_profile_stories


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "story_benchmark_24_v2.json"
QUALITY_REFERENCE = ROOT / "evaluation" / "agent_pipeline_v3_quality_reference.json"
INDEX = ROOT / "data" / "index"
REPORT_ROOT = ROOT / "agent_pipeline_v3" / "reports"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result_dir(output: Path | None) -> Path:
    if output is not None:
        path = output.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = REPORT_ROOT / f"benchmark_3_{stamp}_{uuid.uuid4().hex[:6]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _input_hashes() -> dict[str, str]:
    paths = [
        DATASET,
        QUALITY_REFERENCE,
        ROOT / "agent_pipeline_v2/prompts/extractor_v1.txt",
        ROOT / "agent_pipeline_v2_1/prompts/judge_v2_1.txt",
        ROOT / "agent_pipeline_v2_2/systems.py",
        ROOT / "agent_pipeline_v3/model.py",
        ROOT / "agent_pipeline_v3/profiling.py",
        ROOT / "agent_pipeline_v3/metrics.py",
        ROOT / "agent_pipeline_v3/runner.py",
        INDEX / "CURRENT",
    ]
    current = (INDEX / "CURRENT").read_text(encoding="ascii").strip()
    paths += [INDEX / "versions" / current / "manifest.json", INDEX / "versions" / current / "metadata.json"]
    values = {str(path.relative_to(ROOT)).replace("\\", "/"): _sha(path) for path in paths}
    values["model_id"] = MODEL
    values["model_revision"] = REVISION
    return values


def validate_inputs() -> dict:
    data = _read(DATASET)
    cases = data.get("cases", [])
    ids = [case.get("id") for case in cases]
    reference = _read(QUALITY_REFERENCE)
    if len(cases) != 24 or len(ids) != len(set(ids)) or any(not case_id for case_id in ids):
        raise ValueError("Benchmark 3 requires the fixed 24-story dataset with unique IDs")
    if len(reference.get("cases", {})) != 24:
        raise ValueError("Benchmark 3 requires the fixed 24-case quality reference")
    if not (INDEX / "CURRENT").exists():
        raise ValueError("index is missing")
    return {
        "status": "ok",
        "cases": len(cases),
        "gold_facts": sum(len(case.get("gold_facts", [])) for case in cases),
        "quality_reference_cases": len(reference["cases"]),
        "model_id": MODEL,
        "model_revision": REVISION,
        "judge_batch_size": 8,
        "warmup_stories": 1,
        "repeats": 1,
        "inputs": _input_hashes(),
    }


def _load_runtime(device: str):
    from agent_pipeline_v2_1.retrieval import V21Retriever
    from agent_pipeline_v2_2.systems import build_candidate_system
    from encoder import Encoder
    from retrieval import Retriever

    llm = ProfiledV2QwenJudge(device)
    dense = Retriever(INDEX, Encoder(device=llm.device, offline=True, precision="float32"))
    metadata = _read(ROOT / "evaluation" / "agent_pipeline_v2_1_lore_metadata.json")
    hybrid = V21Retriever(dense, metadata)
    system = build_candidate_system(llm, hybrid, llm.device)
    return llm, system


def _progress(value: dict) -> None:
    kind = value.get("kind")
    if kind == "warmup":
        print(f"[WARMUP] {value['case_id']}", flush=True)
    elif kind == "story":
        print(f"[STORY {value['completed'] + 1}/24] {value['case_id']}", flush=True)
    elif kind == "stage":
        purpose = value.get("purpose", "stage")
        window = value.get("window_id", "")
        print(f"  [{value['case_id']}] {purpose} {window}", flush=True)


def _load_calls(rows: list[dict]) -> list[dict]:
    return [call for row in rows for call in row.get("calls", [])]


def summarize(directory: Path) -> dict:
    rows = load_story_rows(directory / "story_runs.jsonl")
    calls = _load_calls(rows)
    metadata_path = directory / "run_metadata.json"
    metadata = _read(metadata_path) if metadata_path.exists() else {"model_load_sessions": []}
    model_load_seconds = sum(float(value) for value in metadata.get("model_load_sessions", []))
    profile = aggregate_profile(calls, rows, model_load_seconds)
    quality_guard = compare_quality(rows, _read(directory / "quality_reference.json"))
    write_profile_outputs(directory, profile, calls, quality_guard)
    return {"profile": profile, "quality_guard": quality_guard, "rows": rows}


def run(args) -> int:
    validation = validate_inputs()
    data = _read(DATASET)
    directory = _result_dir(args.output)
    manifest = build_manifest(dataset=data, inputs=validation["inputs"], device=args.device)
    ensure_manifest(directory / "manifest.json", manifest)
    if not (directory / "story_dataset.json").exists():
        shutil.copy2(DATASET, directory / "story_dataset.json")
    if not (directory / "quality_reference.json").exists():
        shutil.copy2(QUALITY_REFERENCE, directory / "quality_reference.json")
    print("RESULT_DIR=" + str(directory), flush=True)
    rows = load_story_rows(directory / "story_runs.jsonl")
    if len(rows) < len(data["cases"]):
        print("正在加载一次共享模型和检索器……", flush=True)
        llm, system = _load_runtime(args.device)
        metadata_path = directory / "run_metadata.json"
        metadata = _read(metadata_path) if metadata_path.exists() else {"model_load_sessions": []}
        metadata.setdefault("model_load_sessions", []).append(float(llm.load_seconds))
        atomic_write_json(metadata_path, metadata)
        rows = run_profile_stories(data["cases"], llm, system, directory, process=process_story, progress=_progress)
    result = summarize(directory)
    print("SUMMARY=" + str(directory / "profile_summary.md"), flush=True)
    complete = len(rows) == 24 and all(row.get("status") == "ok" for row in rows)
    return 0 if complete and result["quality_guard"]["comparable"] else 2


def validate_command(args) -> int:
    print(json.dumps(validate_inputs(), ensure_ascii=False, indent=2))
    return 0


def summary_command(args) -> int:
    result = summarize(args.result_dir.resolve())
    print("SUMMARY=" + str(args.result_dir.resolve() / "profile_summary.md"), flush=True)
    return 0 if result["quality_guard"]["comparable"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark 3 LLM workload profiling")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.set_defaults(handler=validate_command)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run_parser.add_argument("--output", type=Path)
    run_parser.set_defaults(handler=run)
    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("--result-dir", type=Path, required=True)
    summary_parser.set_defaults(handler=summary_command)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print("错误：" + str(exc), flush=True)
        return 2
