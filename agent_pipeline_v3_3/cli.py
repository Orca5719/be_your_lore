"""Standalone Benchmark 3.3 base run, before A/B behavior experiments."""

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
from agent_pipeline_v3 import cli as v3_cli
from agent_pipeline_v3.runner import build_manifest, ensure_manifest, load_story_rows, run_profile_stories

from .systems import build_system


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "agent_pipeline_v3_3" / "reports"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs() -> dict:
    baseline = v3_cli.validate_inputs()
    inputs = dict(baseline["inputs"])
    for relative in ("agent_pipeline_v2/extractor.py", "agent_pipeline_v3_3/coverage.py",
                     "agent_pipeline_v3_3/extractor.py", "agent_pipeline_v3_3/systems.py",
                     "agent_pipeline_v3_3/cli.py"):
        inputs[relative] = _sha(ROOT / relative)
    return {**baseline, "stage": "3.3-base", "extractor": "benchmark-3-restored", "inputs": inputs}


def _load_runtime(device: str):
    from agent_pipeline_v2_1.retrieval import V21Retriever
    from agent_pipeline_v3.model import ProfiledV2QwenJudge
    from encoder import Encoder
    from retrieval import Retriever

    llm = ProfiledV2QwenJudge(device)
    dense = Retriever(v3_cli.INDEX, Encoder(device=llm.device, offline=True, precision="float32"))
    metadata = json.loads((ROOT / "evaluation" / "agent_pipeline_v2_1_lore_metadata.json").read_text(encoding="utf-8"))
    hybrid = V21Retriever(dense, metadata)
    return llm, build_system(llm, hybrid, llm.device)


def run(args) -> int:
    validation = validate_inputs()
    dataset = json.loads(v3_cli.DATASET.read_text(encoding="utf-8"))
    if args.output:
        directory = args.output.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = REPORT_ROOT / f"benchmark_3_3_base_{stamp}_{uuid.uuid4().hex[:6]}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(dataset=dataset, inputs=validation["inputs"], device=args.device)
    ensure_manifest(directory / "manifest.json", manifest)
    for source, name in ((v3_cli.DATASET, "story_dataset.json"),
                         (v3_cli.QUALITY_REFERENCE, "quality_reference.json")):
        if not (directory / name).exists():
            shutil.copy2(source, directory / name)
    print("RESULT_DIR=" + str(directory), flush=True)
    rows = load_story_rows(directory / "story_runs.jsonl")
    if len(rows) < len(dataset["cases"]):
        llm, system = _load_runtime(args.device)
        metadata_path = directory / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {"model_load_sessions": []}
        metadata["model_load_sessions"].append(float(llm.load_seconds))
        atomic_write_json(metadata_path, metadata)
        rows = run_profile_stories(dataset["cases"], llm, system, directory,
                                   process=process_story, progress=v3_cli._progress)
    result = v3_cli.summarize(directory)
    print("SUMMARY=" + str(directory / "profile_summary.md"), flush=True)
    complete = len(rows) == 24 and all(row.get("status") == "ok" for row in rows)
    return 0 if complete and result["quality_guard"]["comparable"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark 3.3 restored baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.set_defaults(handler=lambda args: print(json.dumps(validate_inputs(), ensure_ascii=False, indent=2)) or 0)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run_parser.add_argument("--output", type=Path)
    run_parser.set_defaults(handler=run)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print("错误：" + str(exc), flush=True)
        return 2
