"""CLI orchestration for Agent Pipeline v2 quality and batching benchmarks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from .benchmark import aggregate_repeats, benchmark_markdown, score_story_quality
from .fixtures import validate_fixture
from .story_dataset import upgrade_legacy_story_dataset, validate_story_dataset
from .review import build_review_template, propose_review, validate_review
from .summary import build_benchmark_summary, render_benchmark_markdown
from .attribution import build_error_attribution


ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "evaluation" / "agent_pipeline_v2_judge_48.json"
STORIES = ROOT / "evaluation" / "story_benchmark_24_v2.json"
INDEX = ROOT / "data" / "index"
BATCH_SIZES = (1, 2, 4, 8)
FORMAL_BATCH_SIZE = 8
FORMAL_REPEATS = 3


def validate_formal_batch_sizes(batch_sizes) -> list[int]:
    values = list(batch_sizes)
    if values != [FORMAL_BATCH_SIZE]:
        raise ValueError("正式Judge benchmark只允许batch=8；历史batch 1/2/4/8结果只能离线读取")
    return values


def summarize_run_status(story_returncode: int, judge_returncode: int) -> dict:
    tracks = {
        "stories": "ok" if story_returncode == 0 else "failed",
        "judge": "ok" if judge_returncode == 0 else "failed",
    }
    exit_code = 0 if all(status == "ok" for status in tracks.values()) else 2
    return {"status": "ok" if exit_code == 0 else "partial", "tracks": tracks, "exit_code": exit_code}


def story_run_exit_code(rows: list[dict], planned_cases: int) -> int:
    """Partial model outputs are benchmark observations, not execution failures."""
    if len(rows) != planned_cases:
        return 2
    return 2 if any(row.get("result", {}).get("status") == "error" for row in rows) else 0


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result_directory(output: Path | None) -> Path:
    if output is not None:
        directory = output.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = ROOT / "agent_pipeline_v2" / "reports" / f"agent_pipeline_v2_benchmark_{stamp}_{uuid.uuid4().hex[:6]}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def validate_inputs() -> dict:
    fixture = _read(FIXTURE)
    validate_fixture(fixture, expected_per_label=16, expected_k=5)
    stories = _read(STORIES)
    validation_data = upgrade_legacy_story_dataset(stories)
    story_validation = validate_story_dataset(
        validation_data,
        ROOT,
        INDEX,
        expected_cases=24,
        expected_facts_per_verdict=24,
    )
    version = (INDEX / "CURRENT").read_text(encoding="utf-8").strip()
    metadata = _read(INDEX / "versions" / version / "metadata.json")
    if fixture.get("index_version") != version:
        raise ValueError("Judge fixture与当前索引版本不匹配，须重建fixture")
    return {
        "status": "ok",
        "judge_fixture_items": len(fixture["items"]),
        "judge_fixture_digest": fixture["items_digest"],
        "story_cases": len(stories["cases"]),
        "story_gold_facts": sum(len(case["gold_facts"]) for case in stories["cases"]),
        "index_version": version,
        "index_chunks": len(metadata["chunks"]),
        "batch_sizes": [FORMAL_BATCH_SIZE],
        "repeats": FORMAL_REPEATS,
        "story_validation": story_validation,
    }


def run_judge(args) -> int:
    validate_inputs()
    directory = _result_directory(args.output)
    target_fixture = directory / "judge_fixture.json"
    if not target_fixture.exists():
        shutil.copy2(FIXTURE, target_fixture)
    validate_formal_batch_sizes(args.batch_sizes)
    aggregates = []
    return_codes = []
    for batch_size in args.batch_sizes:
        runs = []
        for repeat in range(1, args.repeats + 1):
            output = directory / f"batch_{batch_size}_repeat_{repeat}.json"
            if output.exists():
                raise FileExistsError(f"结果文件已存在：{output}")
            print(f"Judge batch={batch_size} repeat={repeat}/{args.repeats}", flush=True)
            command = [
                sys.executable,
                "-X",
                "utf8",
                "-m",
                "agent_pipeline_v2.benchmark_worker",
                "--fixture",
                str(target_fixture),
                "--output",
                str(output),
                "--batch-size",
                str(batch_size),
                "--repeat",
                str(repeat),
                "--device",
                args.device,
            ]
            completed = subprocess.run(command, cwd=ROOT, check=False)
            return_codes.append(completed.returncode)
            if output.exists():
                runs.append(_read(output))
            else:
                runs.append({"batch_size": batch_size, "repeat": repeat, "status": "error", "oom": False, "error": "worker_did_not_write_result", "metrics": {"facts_total": 0, "judge_accuracy": None}})
        aggregate = aggregate_repeats(runs)
        aggregates.append(aggregate)
        _write(directory / f"batch_{batch_size}.json", aggregate)
    report = {
        "schema_version": "agent-pipeline-v2-batching-benchmark-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if all(row["status"] == "ok" for row in aggregates) else "partial",
        "device": args.device,
        "batch_sizes": list(args.batch_sizes),
        "repeats": args.repeats,
        "fixture_items": 48,
        "results": aggregates,
        "notice": "固定48条Judge输入；Accuracy包含解析失败。Judge计时不含模型加载、fixture构建或retrieval。",
    }
    _write(directory / "judge_report.json", report)
    (directory / "benchmark_summary.md").write_text(benchmark_markdown(aggregates), encoding="utf-8")
    manifest = {
        "fixture_sha256": _sha(target_fixture),
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): _sha(path) for path in (
            ROOT / "agent_pipeline_v2" / "batch_llm.py",
            ROOT / "agent_pipeline_v2" / "judge.py",
            ROOT / "agent_pipeline_v2" / "benchmark.py",
            ROOT / "agent_pipeline_v2" / "benchmark_worker.py",
            ROOT / "agent_pipeline_v2" / "prompts" / "judge_v1.txt",
            ROOT / "agent_pipeline_v2" / "model.py",
            ROOT / "qwen_judge.py",
        )},
    }
    _write(directory / "judge_manifest.json", manifest)
    print("RESULT_DIR=" + str(directory), flush=True)
    return 0 if all(code == 0 for code in return_codes) else 2


def run_stories(args) -> int:
    validate_inputs()
    directory = _result_directory(args.output)
    target_dataset = directory / "story_dataset.json"
    if not target_dataset.exists():
        shutil.copy2(STORIES, target_dataset)
    data = _read(target_dataset)
    from encoder import Encoder
    from .model import MODEL, REVISION, V2QwenJudge
    from retrieval import Retriever
    from .pipeline import process_story

    llm = V2QwenJudge(args.device)
    encoder_started = time.perf_counter()
    retriever = Retriever(INDEX, Encoder(device=llm.device, offline=True, precision="float32"))
    encoder_seconds = time.perf_counter() - encoder_started
    rows = []
    raw_path = directory / "stories_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as stream:
        for number, case in enumerate(data["cases"], 1):
            print(f"Story {number}/{len(data['cases'])}: {case['id']}", flush=True)
            started = time.perf_counter()
            result = process_story(case["story"], llm=llm, retriever=retriever, top_k=args.top_k, judge_batch_size=args.judge_batch_size, device=llm.device,
                progress=lambda step: print("  " + step["purpose"] + " " + str(step.get("window_id", "")), flush=True))
            row = {"case_id": case["id"], "seconds": time.perf_counter() - started, "result": result}
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
    review = build_review_template(data, rows)
    _write(directory / "story_review.json", review)
    _write(directory / "story_review_proposals.json", propose_review(data, rows))
    report = {
        "schema_version": "agent-pipeline-v2-story-run-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending_human_review",
        "model": MODEL,
        "revision": REVISION,
        "device": llm.device,
        "judge_batch_size": args.judge_batch_size,
        "top_k": args.top_k,
        "model_load_seconds": llm.load_seconds,
        "encoder_index_load_seconds": encoder_seconds,
        "case_count": len(rows),
        "case_status_counts": dict(__import__("collections").Counter(row["result"]["status"] for row in rows)),
        "execution_complete": len(rows) == len(data["cases"]),
    }
    _write(directory / "story_report.json", report)
    _write(directory / "story_manifest.json", {"dataset_sha256": _sha(target_dataset), "raw_sha256": _sha(raw_path), "extractor_prompt_sha256": _sha(ROOT / "agent_pipeline_v2" / "prompts" / "extractor_v1.txt")})
    print("RESULT_DIR=" + str(directory), flush=True)
    return story_run_exit_code(rows, len(data["cases"]))


def score_stories(args) -> int:
    directory = args.result_dir.resolve()
    data = _read(directory / "story_dataset.json")
    review = _read(directory / "story_review.json")
    if review.get("status") != "reviewed":
        raise ValueError("story_review.json尚未标为reviewed")
    rows = [json.loads(line) for line in (directory / "stories_raw.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    quality = score_story_quality(data["cases"], rows, review, k=args.top_k)
    attribution = build_error_attribution(data, rows, review, quality)
    _write(directory / "story_quality.json", quality)
    _write(directory / "error_attribution.json", attribution)
    summary = build_benchmark_summary(
        quality,
        attribution,
        _read(directory / "judge_report.json") if (directory / "judge_report.json").exists() else {},
        _read(directory / "run_status.json") if (directory / "run_status.json").exists() else {},
    )
    _write(directory / "benchmark_summary.json", summary)
    (directory / "benchmark_summary.md").write_text(render_benchmark_markdown(summary), encoding="utf-8")
    print(json.dumps(quality, ensure_ascii=False, indent=2))
    return 0


def build_summary(args) -> int:
    directory = args.result_dir.resolve()
    def optional(name, default):
        path = directory / name
        return _read(path) if path.exists() else default
    summary = build_benchmark_summary(
        optional("story_quality.json", {}),
        optional("error_attribution.json", {}),
        optional("judge_report.json", {}),
        optional("run_status.json", {}),
    )
    _write(directory / "benchmark_summary.json", summary)
    (directory / "benchmark_summary.md").write_text(render_benchmark_markdown(summary), encoding="utf-8")
    print("SUMMARY=" + str(directory / "benchmark_summary.json"), flush=True)
    return 0


def run_all(args) -> int:
    validate_formal_batch_sizes(args.batch_sizes)
    if args.repeats < 1:
        raise ValueError("repeats须为正整数")
    directory = _result_directory(args.output)
    script = ROOT / "agent_pipeline_v2_benchmark.py"
    story_command = [sys.executable, "-X", "utf8", str(script), "run-stories", "--device", args.device, "--judge-batch-size", str(args.story_judge_batch_size), "--top-k", str(args.top_k), "--output", str(directory)]
    judge_command = [sys.executable, "-X", "utf8", str(script), "run-judge", "--device", args.device, "--repeats", str(args.repeats), "--output", str(directory), "--batch-sizes", *map(str, args.batch_sizes)]
    story = subprocess.run(story_command, cwd=ROOT, check=False)
    judge = subprocess.run(judge_command, cwd=ROOT, check=False)
    run_status = summarize_run_status(story.returncode, judge.returncode)
    _write(directory / "run_status.json", run_status)
    print("STORY_TRACK=" + run_status["tracks"]["stories"], flush=True)
    print("JUDGE_TRACK=" + run_status["tracks"]["judge"], flush=True)
    print("RUN_STATUS=" + run_status["status"], flush=True)
    print("RESULT_DIR=" + str(directory), flush=True)
    return run_status["exit_code"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    judge = subparsers.add_parser("run-judge")
    judge.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    judge.add_argument("--batch-sizes", nargs="+", type=int, default=[FORMAL_BATCH_SIZE])
    judge.add_argument("--repeats", type=int, default=FORMAL_REPEATS)
    judge.add_argument("--output", type=Path)
    stories = subparsers.add_parser("run-stories")
    stories.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    stories.add_argument("--judge-batch-size", type=int, default=4)
    stories.add_argument("--top-k", type=int, default=5)
    stories.add_argument("--output", type=Path)
    score = subparsers.add_parser("score-stories")
    score.add_argument("--result-dir", type=Path, required=True)
    score.add_argument("--top-k", type=int, default=5)
    summary = subparsers.add_parser("summary")
    summary.add_argument("--result-dir", type=Path, required=True)
    all_parser = subparsers.add_parser("run-all")
    all_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    all_parser.add_argument("--batch-sizes", nargs="+", type=int, default=[FORMAL_BATCH_SIZE])
    all_parser.add_argument("--repeats", type=int, default=FORMAL_REPEATS)
    all_parser.add_argument("--story-judge-batch-size", "--judge-batch-size", dest="story_judge_batch_size", type=int, default=8)
    all_parser.add_argument("--top-k", type=int, default=5)
    all_parser.add_argument("--output", type=Path)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            print(json.dumps(validate_inputs(), ensure_ascii=False))
            return 0
        if args.command == "run-judge":
            if args.repeats < 1:
                raise ValueError("repeats须为正整数")
            validate_formal_batch_sizes(args.batch_sizes)
            return run_judge(args)
        if args.command == "run-stories":
            return run_stories(args)
        if args.command == "score-stories":
            return score_stories(args)
        if args.command == "summary":
            return build_summary(args)
        if args.command == "run-all":
            return run_all(args)
        raise ValueError("未知命令")
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print("错误：" + str(exc))
        return 2
