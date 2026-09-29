from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

from .attribution import attribute_first_failures
from .paired import run_paired_matrix
from .report import build_summary, write_summary
from .review import build_review_template, validate_review
from .runner import aggregate_performance, atomic_write_json, build_manifest, ensure_manifest, load_jsonl, run_resumable, write_jsonl_atomic
from .schema import SystemConfig
from .scoring import score_system
from .story_pipeline import process_story


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evaluation" / "story_benchmark_24_v2.json"
INDEX = ROOT / "data" / "index"
REPORT_ROOT = ROOT / "agent_pipeline_v2_2" / "reports"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result_dir(output: Path | None) -> Path:
    if output:
        path = output.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = REPORT_ROOT / f"benchmark_2_2_{stamp}_{uuid.uuid4().hex[:6]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _input_hashes() -> dict[str, str]:
    paths = [
        DATASET, ROOT / "agent_pipeline_v2/prompts/extractor_v1.txt",
        ROOT / "agent_pipeline_v2/prompts/judge_v1.txt", ROOT / "agent_pipeline_v2_1/prompts/judge_v2_1.txt",
        ROOT / "agent_pipeline_v2_2/systems.py", ROOT / "agent_pipeline_v2_2/paired.py",
        ROOT / "agent_pipeline_v2/extractor.py", ROOT / "agent_pipeline_v2/retrieval.py", ROOT / "agent_pipeline_v2/judge.py",
        ROOT / "agent_pipeline_v2_1/extractor.py", ROOT / "agent_pipeline_v2_1/retrieval.py", ROOT / "agent_pipeline_v2_1/story_pipeline.py",
        INDEX / "CURRENT",
    ]
    current = (INDEX / "CURRENT").read_text(encoding="ascii").strip()
    paths += [INDEX / "versions" / current / "manifest.json", INDEX / "versions" / current / "metadata.json"]
    values = {str(path.relative_to(ROOT)).replace("\\", "/"): _sha(path) for path in paths}
    from qwen_judge import MODEL, REVISION
    values["model_id"] = MODEL
    values["model_revision"] = REVISION
    return values


def validate_inputs() -> dict:
    data = _read(DATASET)
    cases = data.get("cases", [])
    ids = [case.get("id") for case in cases]
    if len(cases) != 24 or len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("Benchmark 2.2 requires the fixed 24-story dataset with unique IDs")
    if not (INDEX / "CURRENT").exists():
        raise ValueError("index is missing")
    return {"status": "ok", "cases": len(cases), "gold_facts": sum(len(case.get("gold_facts", [])) for case in cases), "baseline": SystemConfig.baseline().to_dict(), "candidate": SystemConfig.candidate().to_dict(), "inputs": _input_hashes()}


def _manifest(data: dict, args) -> dict:
    return build_manifest(dataset=data, configs=[SystemConfig.baseline().to_dict(), SystemConfig.candidate().to_dict()], inputs=_input_hashes(), repeats=args.repeats, warmup=args.warmup)


def _load_runtime(device: str):
    from agent_pipeline_v2.model import V2QwenJudge
    from agent_pipeline_v2_1.retrieval import V21Retriever
    from encoder import Encoder
    from retrieval import Retriever
    from .systems import build_baseline_system, build_candidate_system
    llm = V2QwenJudge(device)
    dense = Retriever(INDEX, Encoder(device=llm.device, offline=True, precision="float32"))
    metadata = _read(ROOT / "evaluation" / "agent_pipeline_v2_1_lore_metadata.json")
    hybrid = V21Retriever(dense, metadata)
    return llm, dense, hybrid, build_baseline_system(llm, dense, llm.device), build_candidate_system(llm, hybrid, llm.device)


def run_end_to_end(args, runtime=None) -> int:
    validate_inputs()
    data = _read(DATASET)
    directory = _result_dir(args.output)
    target_dataset = directory / "story_dataset.json"
    if not target_dataset.exists(): shutil.copy2(DATASET, target_dataset)
    manifest = _manifest(data, args)
    ensure_manifest(directory / "manifest.json", manifest)
    print("RESULT_DIR=" + str(directory), flush=True)
    print("端到端轨道：已完成的案例将自动跳过。", flush=True)
    _, _, _, baseline, candidate = runtime or _load_runtime(args.device)
    def on_start(case, system, completed, total):
        print(f"[E2E {completed + 1}/{total}] {case['id']} / {system.config.name}", flush=True)
    def progress_factory(case, system):
        return lambda step: print(f"  [{system.config.name}] {step.get('purpose', 'stage')} {step.get('window_id', '')}", flush=True)
    rows = run_resumable(
        data["cases"], [baseline, candidate], directory / "end_to_end.jsonl", process_story,
        on_start=on_start, progress_factory=progress_factory,
    )
    if not (directory / "review.json").exists():
        atomic_write_json(directory / "review.json", build_review_template(data, rows))
    print("RESULT_DIR=" + str(directory), flush=True)
    return 2 if any(row["status"] == "error" for row in rows) else 0


def _canonical_retrievals(extraction, dense, hybrid, top_k, progress=None):
    from agent_pipeline_v2.retrieval import retrieve_events
    from agent_pipeline_v2_1.story_pipeline import event_fact, retrieve_story_events
    dense_v1 = retrieve_events(extraction, retriever=dense, k=top_k, progress=progress)
    hybrid_v21 = retrieve_story_events(extraction, hybrid, "hybrid", False, top_k, progress)
    def to_v21(v1):
        value = dict(v1)
        value["items"] = [{**item, "fact": event_fact(item["event"])} for item in v1["items"]]
        return value
    def to_v1(v21):
        return {"schema_version": "agent-pipeline-v2-retrieval-v1", "stage": "retrieval", "status": v21["status"], "extraction": v21["extraction"], "events": v21["events"], "items": [{key: item[key] for key in ("event_id", "event", "query", "evidence", "status") if key in item} for item in v21["items"]], "top_k": top_k, "upstream_status": v21["extraction"]["status"], "failure_reasons": v21["failure_reasons"], "retrieval_complete": v21["status"] == "ok", "processing_complete": v21["status"] == "ok", "encoder_loaded_this_request": False, "index": None, "device": "shared"}
    return {"dense": {"v1": dense_v1, "v21": to_v21(dense_v1)}, "hybrid": {"v1": to_v1(hybrid_v21), "v21": hybrid_v21}}


def run_paired(args, runtime=None) -> int:
    directory = args.result_dir.resolve()
    data = _read(directory / "story_dataset.json")
    end_rows = load_jsonl(directory / "end_to_end.jsonl")
    baseline = {row["case_id"]: row for row in end_rows if row["system"] == "baseline"}
    llm, dense, hybrid, _, _ = runtime or _load_runtime(args.device)
    from agent_pipeline_v2.judge import judge_events
    from agent_pipeline_v2_1.story_pipeline import judge_story_events
    paired_path = directory / "paired.jsonl"
    rows = load_jsonl(paired_path) if paired_path.exists() else []
    done = {row["case_id"] for row in rows}
    for case in data["cases"]:
        if case["id"] in done:
            continue
        print(f"[PAIRED {len(rows) + 1}/{len(data['cases'])}] {case['id']}", flush=True)
        extraction = baseline[case["id"]]["result"]["benchmark_stages"]["extraction"]
        progress = lambda step: print(f"  [paired] {step.get('purpose', 'stage')} {step.get('window_id', '')}", flush=True)
        cached = _canonical_retrievals(extraction, dense, hybrid, args.top_k, progress)
        matrix = run_paired_matrix(
            extraction["events"],
            lambda extraction, top_k: cached["dense"], lambda extraction, top_k: cached["hybrid"],
            lambda value, batch_size: judge_events(value["v1"], batch_size=batch_size, llm=llm, progress=progress),
            lambda value, batch_size: judge_story_events(value["v21"], llm, batch_size, progress),
            top_k=args.top_k, batch_size=args.judge_batch_size,
        )
        rows.append({"case_id": case["id"], "matrix": matrix})
        write_jsonl_atomic(paired_path, rows)
    print("RESULT_DIR=" + str(directory), flush=True)
    return 2 if any(row["matrix"]["status"] != "ok" for row in rows) else 0


def _review_for_system(review: dict, system: str) -> dict:
    result = {key: {} for key in ("event_mapping", "finding_mapping", "system_event_labels", "system_finding_labels", "reasoning_support_labels")}
    for section in result:
        for key, value in review.get(section, {}).items():
            prefix = system + ":"
            if key.startswith(prefix): result[section][key[len(prefix):]] = value
    return result


def score(args) -> int:
    directory = args.result_dir.resolve()
    data, rows, review = _read(directory / "story_dataset.json"), load_jsonl(directory / "end_to_end.jsonl"), _read(directory / "review.json")
    validate_review(data, rows, review)
    scores, attributions = {}, {}
    for system in ("baseline", "candidate"):
        system_rows = [row for row in rows if row["system"] == system]
        system_review = _review_for_system(review, system)
        scores[system] = score_system(data["cases"], system_rows, system_review, 5)
        attributions[system] = attribute_first_failures(data["cases"], system_rows, system_review, 5)
    atomic_write_json(directory / "quality.json", scores)
    atomic_write_json(directory / "attribution.json", attributions)
    print("RESULT_DIR=" + str(directory), flush=True)
    return 0


def summary(args) -> int:
    directory = args.result_dir.resolve()
    quality = _read(directory / "quality.json")
    attribution = _read(directory / "attribution.json")
    performance = _read(directory / "performance.json") if (directory / "performance.json").exists() else {}
    report = build_summary(quality["baseline"], quality["candidate"], performance=performance, attribution=attribution, manifest=_read(directory / "manifest.json"))
    write_summary(directory, report)
    print("SUMMARY=" + str(directory / "benchmark_2_2_summary.md"), flush=True)
    return 0


def run_all(args) -> int:
    if args.output is None:
        args.output = _result_dir(None)
    print("RESULT_DIR=" + str(args.output.resolve()), flush=True)
    runtime = _load_runtime(args.device)
    code = run_end_to_end(args, runtime)
    paired_args = argparse.Namespace(**vars(args), result_dir=args.output or Path())
    paired = run_paired(paired_args, runtime)
    data = _read((args.output / "story_dataset.json").resolve())
    _, _, _, baseline, candidate = runtime
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup cannot be negative")
    quality_rows = load_jsonl(args.output.resolve() / "end_to_end.jsonl")
    by_system = defaultdict(list)
    # Warm-up establishes stable kernels/caches; one fixed story per system is sufficient and excluded.
    for warmup in range(1, args.warmup + 1):
        for system in (baseline, candidate):
            path = args.output.resolve() / f"performance_{system.config.name}_warmup_{warmup}.jsonl"
            rows = load_jsonl(path)
            if not rows:
                case = data["cases"][0]
                print(f"[PERF warmup {warmup}/{args.warmup}] {system.config.name} / {case['id']}", flush=True)
                rows.append(process_story(system, case["id"], case["story"], progress=lambda step: print(f"  [{system.config.name}] {step.get('purpose', 'stage')} {step.get('window_id', '')}", flush=True)))
                write_jsonl_atomic(path, rows)
            by_system[system.config.name].append(rows)
    # The deterministic quality run is also measured repeat 1; do not execute it twice.
    for system in (baseline, candidate):
        by_system[system.config.name].append([row for row in quality_rows if row["system"] == system.config.name])
    for repeat in range(2, args.repeats + 1):
        for system in (baseline, candidate):
            path = args.output.resolve() / f"performance_{system.config.name}_repeat_{repeat}.jsonl"
            rows = load_jsonl(path)
            done = {row["case_id"] for row in rows}
            for number, case in enumerate(data["cases"], 1):
                if case["id"] in done:
                    continue
                print(f"[PERF repeat {repeat}/{args.repeats} {number}/{len(data['cases'])}] {system.config.name} / {case['id']}", flush=True)
                rows.append(process_story(system, case["id"], case["story"], progress=lambda step, name=system.config.name: print(f"  [{name}] {step.get('purpose', 'stage')} {step.get('window_id', '')}", flush=True)))
                write_jsonl_atomic(path, rows)
            by_system[system.config.name].append(rows)
    performance = {name: aggregate_performance(runs, args.warmup) for name, runs in by_system.items()}
    atomic_write_json(args.output.resolve() / "performance.json", performance)
    print("RESULT_DIR=" + str(args.output.resolve()), flush=True)
    return 2 if code or paired else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent Pipeline Benchmark 2.2")
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("validate")
    def common(p):
        p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
        p.add_argument("--top-k", type=int, default=5)
        p.add_argument("--judge-batch-size", type=int, default=8)
        p.add_argument("--repeats", type=int, default=3)
        p.add_argument("--warmup", type=int, default=1)
    end = subs.add_parser("run-end-to-end"); common(end); end.add_argument("--output", type=Path)
    paired = subs.add_parser("run-paired"); common(paired); paired.add_argument("--result-dir", type=Path, required=True)
    score_p = subs.add_parser("score"); score_p.add_argument("--result-dir", type=Path, required=True)
    summary_p = subs.add_parser("summary"); summary_p.add_argument("--result-dir", type=Path, required=True)
    all_p = subs.add_parser("run-all"); common(all_p); all_p.add_argument("--output", type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate": print(json.dumps(validate_inputs(), ensure_ascii=False, indent=2)); return 0
        if args.command == "run-end-to-end": return run_end_to_end(args)
        if args.command == "run-paired": return run_paired(args)
        if args.command == "score": return score(args)
        if args.command == "summary": return summary(args)
        if args.command == "run-all": return run_all(args)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print("错误：" + str(exc), file=sys.stderr)
        return 2
    return 2
