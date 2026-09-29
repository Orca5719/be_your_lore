"""Small independent CLI surface for Agent Pipeline 2.1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
import uuid
import time
from collections import Counter

from . import VERSION
from .contracts import BenchmarkConfig
from .oracle_benchmark import build_comparison_report, render_comparison_markdown
from .oracle_fixture import build_oracle_fixture
from .lore_metadata import build_lore_metadata, validate_lore_metadata
from .metadata_filter import audit_oracle_fixture
from .retrieval_benchmark import render_retrieval_markdown
from agent_pipeline_v2.benchmark import aggregate_repeats


ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent_pipeline_v2_1_benchmark.py")
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser("info", help="显示2.1版本与实施状态")
    info.add_argument("--json", action="store_true", dest="as_json")
    build = commands.add_parser("build-oracle-fixture", help="构建72条Oracle Judge输入")
    build.add_argument("--output", type=Path, default=ROOT / "evaluation" / "agent_pipeline_v2_1_oracle_72.json")
    metadata = commands.add_parser("build-lore-metadata", help="为冻结索引生成无LLM的规则metadata")
    metadata.add_argument("--fixture", type=Path, default=ROOT / "evaluation" / "agent_pipeline_v2_1_oracle_72.json")
    metadata.add_argument("--output", type=Path, default=ROOT / "evaluation" / "agent_pipeline_v2_1_lore_metadata.json")
    audit = commands.add_parser("audit-metadata-filter", help="用72条Oracle事实审计metadata候选池召回")
    audit.add_argument("--fixture", type=Path, default=ROOT / "evaluation" / "agent_pipeline_v2_1_oracle_72.json")
    audit.add_argument("--metadata", type=Path, default=ROOT / "evaluation" / "agent_pipeline_v2_1_lore_metadata.json")
    audit.add_argument("--top-k", type=int, default=5)
    audit.add_argument("--min-candidates", type=int)
    audit.add_argument("--output", type=Path, default=ROOT / "evaluation" / "agent_pipeline_v2_1_metadata_filter_audit.json")
    retrieval = commands.add_parser("run-retrieval", help="运行Benchmark 2.1B六组检索对照")
    retrieval.add_argument("--fixture", type=Path, default=ROOT / "evaluation" / "agent_pipeline_v2_1_oracle_72.json")
    retrieval.add_argument("--metadata", type=Path, default=ROOT / "evaluation" / "agent_pipeline_v2_1_lore_metadata.json")
    retrieval.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    retrieval.add_argument("--top-k", type=int, default=5)
    retrieval.add_argument("--repeats", type=int, default=3)
    retrieval.add_argument("--query-batch-size", type=int, default=16)
    retrieval.add_argument("--output", type=Path)
    stories = commands.add_parser("run-stories", help="运行24篇故事的v2.1端到端诊断")
    stories.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    stories.add_argument("--method", choices=("dense", "bm25", "hybrid"), default="hybrid")
    stories.add_argument("--metadata-filter", action=argparse.BooleanOptionalAction, default=False)
    stories.add_argument("--top-k", type=int, default=5)
    stories.add_argument("--judge-batch-size", type=int, default=8)
    stories.add_argument("--reuse-ok-from", type=Path, help="复用兼容目录中status=ok的故事，只重跑partial/error")
    stories.add_argument("--output", type=Path)
    score = commands.add_parser("score-stories", help="使用已审核账本计算v2.1端到端质量")
    score.add_argument("--result-dir", type=Path, required=True)
    score.add_argument("--top-k", type=int, default=5)
    oracle = commands.add_parser("run-oracle-judge", help="运行Benchmark 2.1A：Oracle Judge v1/v2.1对照")
    oracle.add_argument("--fixture", type=Path, default=ROOT / "evaluation" / "agent_pipeline_v2_1_oracle_72.json")
    oracle.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    oracle.add_argument("--batch-size", type=int, default=8)
    oracle.add_argument("--repeats", type=int, default=3)
    oracle.add_argument("--prompt-versions", nargs="+", choices=("v1", "v2.1"), default=["v1", "v2.1"])
    oracle.add_argument("--reuse-v1-from", type=Path, help="复用一次兼容且已完成的v1结果目录，只重跑v2.1")
    oracle.add_argument("--output", type=Path)
    return parser


def benchmark_info() -> dict:
    config = BenchmarkConfig()
    return {
        "schema_version": "agent-pipeline-v2.1-benchmark-info-v1",
        "package": "agent_pipeline_v2_1",
        "version": VERSION,
        "judge_batch_size": config.judge_batch_size,
        "repeats": config.repeats,
        "implemented_stages": ["scaffold", "judge_contract", "oracle_fixture", "oracle_benchmark", "lore_metadata", "metadata_filter", "bm25", "rrf_hybrid", "retrieval_benchmark_2_1B", "story_diagnostic_24"],
        "planned_benchmarks": ["2.1A", "2.1B", "2.1C"],
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "info":
        value = benchmark_info()
        print(json.dumps(value, ensure_ascii=False, indent=2) if args.as_json else "Agent Pipeline 2.1 scaffold ready")
        return 0
    if args.command == "build-oracle-fixture":
        dataset = json.loads((ROOT / "evaluation" / "story_benchmark_24_v2.json").read_text(encoding="utf-8"))
        version = dataset["corpus_snapshot"]["index_version"]
        metadata = json.loads((ROOT / "data" / "index" / "versions" / version / "metadata.json").read_text(encoding="utf-8"))
        curation = json.loads((ROOT / "evaluation" / "agent_pipeline_v2_1_uncertain_oracle_lore.json").read_text(encoding="utf-8"))
        fixture = build_oracle_fixture(dataset, metadata, curation)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("ORACLE_FIXTURE=" + str(args.output), flush=True)
        return 0
    if args.command == "build-lore-metadata":
        fixture = json.loads(args.fixture.read_text(encoding="utf-8-sig"))
        version = fixture["index_version"]
        source = ROOT / "data" / "index" / "versions" / version / "metadata.json"
        index_metadata = json.loads(source.read_text(encoding="utf-8-sig"))
        chunks = index_metadata["chunks"]
        value = build_lore_metadata(chunks, version)
        validate_lore_metadata(value, chunks, version)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
        print("LORE_METADATA=" + str(args.output), flush=True)
        print(json.dumps(value["summary"], ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "audit-metadata-filter":
        fixture = json.loads(args.fixture.read_text(encoding="utf-8-sig"))
        metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
        value = audit_oracle_fixture(fixture, metadata, k=args.top_k, min_candidates=args.min_candidates)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
        print("METADATA_FILTER_AUDIT=" + str(args.output), flush=True)
        print(json.dumps({key: value[key] for key in ("facts_total", "oracle_lore_pool_hits", "oracle_lore_pool_recall", "selected_level_counts", "candidate_count")}, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "run-retrieval":
        return run_retrieval_benchmark(args)
    if args.command == "run-stories":
        return run_stories_v21(args)
    if args.command == "score-stories":
        return score_stories_v21(args)
    if args.command == "run-oracle-judge":
        return run_oracle_benchmark(args)
    raise ValueError("未知命令")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result_dir(output: Path | None) -> Path:
    if output is not None:
        return output.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "agent_pipeline_v2_1" / "reports" / f"benchmark_2_1A_{stamp}_{uuid.uuid4().hex[:6]}"


def _retrieval_result_dir(output: Path | None) -> Path:
    if output is not None:
        return output.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "agent_pipeline_v2_1" / "reports" / f"benchmark_2_1B_{stamp}_{uuid.uuid4().hex[:6]}"


def _story_result_dir(output: Path | None) -> Path:
    if output is not None:
        return output.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "agent_pipeline_v2_1" / "reports" / f"story_diagnostic_2_1_{stamp}_{uuid.uuid4().hex[:6]}"


def run_retrieval_benchmark(args) -> int:
    from .retrieval_worker import run_retrieval_worker

    result_dir = _retrieval_result_dir(args.output)
    result_dir.mkdir(parents=True, exist_ok=False)
    report = run_retrieval_worker(ROOT, args.fixture.resolve(), args.metadata.resolve(), args.device, args.top_k, args.repeats, args.query_batch_size)
    (result_dir / "retrieval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (result_dir / "retrieval_report.md").write_text(render_retrieval_markdown(report), encoding="utf-8")
    for run in report["runs"]:
        (result_dir / f"retrieval_repeat_{run['repeat']}.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "agent-pipeline-v2.1-retrieval-manifest-v1",
        "benchmark": "2.1B",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fixture_sha256": _sha256(args.fixture.resolve()),
        "lore_metadata_sha256": _sha256(args.metadata.resolve()),
        "top_k": args.top_k,
        "repeats": args.repeats,
        "query_batch_size": args.query_batch_size,
        "device_requested": args.device,
        "config_ids": [row["config_id"] for row in report["results"]],
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("RESULT_DIR=" + str(result_dir), flush=True)
    return 0 if report["status"] == "ok" else 2


def run_stories_v21(args) -> int:
    from agent_pipeline_v2.model import MODEL, REVISION, V2QwenJudge
    from agent_pipeline_v2.review import build_review_template, propose_review
    from encoder import Encoder
    from retrieval import Retriever
    from .retrieval import V21Retriever
    from .story_pipeline import process_story_v21

    dataset_path = ROOT / "evaluation" / "story_benchmark_24_v2.json"
    metadata_path = ROOT / "evaluation" / "agent_pipeline_v2_1_lore_metadata.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
    lore_metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if len(dataset.get("cases", [])) != 24:
        raise ValueError("v2.1端到端诊断要求24篇故事")
    directory = _story_result_dir(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(dataset_path, directory / "story_dataset.json")
    reused_rows = {}
    if args.reuse_ok_from is not None:
        source_dir = args.reuse_ok_from.resolve()
        source_report = json.loads((source_dir / "story_report.json").read_text(encoding="utf-8-sig"))
        source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8-sig"))
        if source_manifest.get("dataset_sha256") != _sha256(dataset_path):
            raise ValueError("复用目录的故事数据集不一致")
        expected = {"method": args.method, "metadata_filter": args.metadata_filter, "top_k": args.top_k, "judge_batch_size": args.judge_batch_size}
        if any(source_report.get(key) != value for key, value in expected.items()):
            raise ValueError("复用目录的检索或Judge配置不一致")
        old_rows = [json.loads(line) for line in (source_dir / "stories_raw.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        reused_rows = {row["case_id"]: row for row in old_rows if row.get("result", {}).get("status") == "ok"}
    llm = V2QwenJudge(args.device)
    encoder_started = time.perf_counter()
    dense = Retriever(ROOT / "data" / "index", Encoder(device=llm.device, offline=True, precision="float32"))
    retriever = V21Retriever(dense, lore_metadata)
    encoder_seconds = time.perf_counter() - encoder_started
    rows = []
    raw_path = directory / "stories_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as stream:
        for number, case in enumerate(dataset["cases"], 1):
            if case["id"] in reused_rows:
                row = reused_rows[case["id"]]
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"Story {number}/24: {case['id']} reused", flush=True)
                continue
            print(f"Story {number}/24: {case['id']}", flush=True)
            started = time.perf_counter()
            result = process_story_v21(
                case["story"], llm, retriever, method=args.method,
                metadata_filter=args.metadata_filter, top_k=args.top_k,
                judge_batch_size=args.judge_batch_size, device=llm.device,
                progress=lambda step: print("  " + step["purpose"] + " " + str(step.get("window_id", "")), flush=True),
            )
            row = {"case_id": case["id"], "seconds": time.perf_counter() - started, "result": result}
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
    review = build_review_template(dataset, rows)
    (directory / "story_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    proposals = propose_review(dataset, rows)
    (directory / "story_review_proposals.json").write_text(json.dumps(proposals, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    statuses = Counter(row["result"]["status"] for row in rows)
    report = {
        "schema_version": "agent-pipeline-v2.1-story-diagnostic-v1",
        "status": "pending_human_review",
        "execution_complete": len(rows) == 24,
        "case_count": len(rows),
        "case_status_counts": dict(statuses),
        "method": args.method,
        "metadata_filter": args.metadata_filter,
        "top_k": args.top_k,
        "judge_batch_size": args.judge_batch_size,
        "model": MODEL,
        "revision": REVISION,
        "device": llm.device,
        "model_load_seconds": llm.load_seconds,
        "encoder_index_load_seconds": encoder_seconds,
        "total_story_seconds": sum(row["seconds"] for row in rows),
        "reused_case_count": len(reused_rows),
        "rerun_case_count": len(rows) - len(reused_rows),
    }
    (directory / "story_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "dataset_sha256": _sha256(directory / "story_dataset.json"),
        "raw_sha256": _sha256(raw_path),
        "lore_metadata_sha256": _sha256(metadata_path),
        "extractor_prompt_sha256": _sha256(ROOT / "agent_pipeline_v2" / "prompts" / "extractor_v1.txt"),
        "judge_prompt_sha256": _sha256(ROOT / "agent_pipeline_v2_1" / "prompts" / "judge_v2_1.txt"),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("RESULT_DIR=" + str(directory), flush=True)
    return 2 if len(rows) != 24 or any(row["result"]["status"] == "error" for row in rows) else 0


def score_stories_v21(args) -> int:
    from agent_pipeline_v2.benchmark import score_story_quality
    from agent_pipeline_v2.review import validate_review

    directory = args.result_dir.resolve()
    dataset = json.loads((directory / "story_dataset.json").read_text(encoding="utf-8-sig"))
    review = json.loads((directory / "story_review.json").read_text(encoding="utf-8-sig"))
    rows = [json.loads(line) for line in (directory / "stories_raw.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_review(dataset, rows, review, require_complete=True)
    quality = score_story_quality(dataset["cases"], rows, review, k=args.top_k)
    (directory / "story_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    extraction = quality["recall_extraction"]
    retrieval = quality["recall_retrieval_at_k"]
    judge = quality["accuracy_judge"]
    conflict = quality["end_to_end_conflict"]
    def percent(value):
        return "N/A" if value is None else f"{value:.2%}"
    lines = [
        "# Agent Pipeline 2.1 — 24 Story Diagnostic", "",
        "| Metric | Value |", "|---|---:|",
        f"| Recall Extraction | {extraction['recall']:.2%} |",
        f"| Recall Retrieval@{retrieval['k']} | {retrieval['recall']:.2%} |",
        f"| Judge Accuracy | {percent(judge['accuracy'])} |",
        f"| Conflict Precision | {percent(conflict['precision'])} |",
        f"| Conflict Recall | {percent(conflict['recall'])} |",
        f"| Conflict F1 | {percent(conflict['f1'])} |", "",
    ]
    (directory / "story_quality.md").write_text("\n".join(lines), encoding="utf-8")
    print("STORY_QUALITY=" + str(directory / "story_quality.json"), flush=True)
    return 0


def run_oracle_benchmark(args) -> int:
    if args.batch_size < 1 or args.repeats < 1:
        raise ValueError("batch-size和repeats必须为正整数")
    fixture = args.fixture.resolve()
    payload = json.loads(fixture.read_text(encoding="utf-8-sig"))
    if payload.get("status") != "reviewed" or len(payload.get("items", [])) != 72:
        raise ValueError("Benchmark 2.1A要求reviewed状态的72条Oracle fixture")
    result_dir = _result_dir(args.output)
    result_dir.mkdir(parents=True, exist_ok=False)
    fixture_copy = result_dir / "oracle_fixture_72.json"
    shutil.copy2(fixture, fixture_copy)
    fixture_sha = _sha256(fixture_copy)
    aggregates = []
    worker_failures = []
    for prompt_version in args.prompt_versions:
        runs = []
        safe_version = prompt_version.replace(".", "_")
        if prompt_version == "v1" and args.reuse_v1_from is not None:
            source_dir = args.reuse_v1_from.resolve()
            source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
            if source_manifest.get("fixture_sha256") != fixture_sha:
                raise ValueError("复用目录的Oracle fixture与当前fixture不一致")
            if source_manifest.get("files", {}).get("judge_v1_prompt") != _sha256(ROOT / "agent_pipeline_v2" / "prompts" / "judge_v1.txt"):
                raise ValueError("复用目录的Judge v1提示词版本不一致")
            for repeat in range(1, args.repeats + 1):
                source = source_dir / f"oracle_v1_repeat_{repeat}.json"
                if not source.exists():
                    raise ValueError(f"复用目录缺少v1第{repeat}次结果")
                target = result_dir / source.name
                shutil.copy2(source, target)
                runs.append(json.loads(target.read_text(encoding="utf-8")))
            print(f"已复用 Judge v1 的 {args.repeats} 次结果。", flush=True)
        else:
            for repeat in range(1, args.repeats + 1):
                output = result_dir / f"oracle_{safe_version}_repeat_{repeat}.json"
                command = [
                    sys.executable, "-X", "utf8", "-m", "agent_pipeline_v2_1.oracle_worker",
                    "--fixture", str(fixture_copy), "--output", str(output),
                    "--prompt-version", prompt_version, "--batch-size", str(args.batch_size),
                    "--repeat", str(repeat), "--device", args.device,
                ]
                print(f"运行 Judge {prompt_version}，第 {repeat}/{args.repeats} 次…", flush=True)
                completed = subprocess.run(command, cwd=ROOT, check=False)
                if not output.exists():
                    raise RuntimeError(f"worker未生成结果：{output}")
                run = json.loads(output.read_text(encoding="utf-8"))
                runs.append(run)
                if completed.returncode != 0:
                    worker_failures.append({"prompt_version": prompt_version, "repeat": repeat, "exit_code": completed.returncode})
        aggregate = aggregate_repeats(runs)
        aggregate.update(prompt_version=prompt_version, fixture_sha256=fixture_sha)
        aggregate_path = result_dir / f"oracle_{safe_version}.json"
        aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        aggregates.append(aggregate)
    if set(args.prompt_versions) == {"v1", "v2.1"}:
        report = build_comparison_report(aggregates, fixture_sha)
        report["worker_failures"] = worker_failures
        (result_dir / "oracle_judge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (result_dir / "oracle_judge_report.md").write_text(render_comparison_markdown(report), encoding="utf-8")
    manifest = {
        "schema_version": "agent-pipeline-v2.1-oracle-manifest-v1",
        "benchmark": "2.1A",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fixture_sha256": fixture_sha,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "prompt_versions": args.prompt_versions,
        "device_requested": args.device,
        "files": {
            "judge_v1_prompt": _sha256(ROOT / "agent_pipeline_v2" / "prompts" / "judge_v1.txt"),
            "judge_v2_1_prompt": _sha256(ROOT / "agent_pipeline_v2_1" / "prompts" / "judge_v2_1.txt"),
            "oracle_benchmark": _sha256(ROOT / "agent_pipeline_v2_1" / "oracle_benchmark.py"),
            "oracle_worker": _sha256(ROOT / "agent_pipeline_v2_1" / "oracle_worker.py"),
        },
        "worker_failures": worker_failures,
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("RESULT_DIR=" + str(result_dir), flush=True)
    return 2 if worker_failures or any(item.get("status") == "error" for item in aggregates) else 0
