"""Benchmark 4.1: score saved 4B extraction, then reuse frozen B3 downstream stages."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

from agent_pipeline_v2_2.runner import atomic_write_json, write_jsonl_atomic

from .audit import score_end_to_end, score_extraction, validate_labels


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation/story_benchmark_24_v2.json"
SOURCE = ROOT / "agent_pipeline_v4/reports/benchmark_4B_20261004T132545Z_0fdce1"
REVIEW = ROOT / "agent_pipeline_v4_1/review.json"
REPORTS = ROOT / "agent_pipeline_v4_1/reports"
CAPACITIES = (2, 4, 8)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_rows(capacity: int):
    return _rows(SOURCE / f"capacity_{capacity}" / "extractor_runs.jsonl")


def inputs():
    from agent_pipeline_v4.cli import validate_inputs

    baseline = validate_inputs()  # Checks B3 dataset, model revision and frozen extractor files.
    cases = baseline["cases"]
    case_ids = {case["id"] for case in cases}
    review = _read(REVIEW)
    if review.get("schema_version") != "benchmark-4.1-review-v1" or review.get("provenance") != "assistant-reviewed":
        raise ValueError("4.1 requires the completed, provenance-labelled review ledger")
    if review.get("dataset_sha256") != _sha(DATASET):
        raise ValueError("review dataset digest mismatch")
    for capacity in CAPACITIES:
        path = SOURCE / f"capacity_{capacity}" / "extractor_runs.jsonl"
        if review.get("source_sha256", {}).get(str(capacity)) != _sha(path):
            raise ValueError(f"CB{capacity} extraction digest mismatch")
        rows = _source_rows(capacity)
        if {row["case_id"] for row in rows} != case_ids:
            raise ValueError(f"CB{capacity} story IDs mismatch")
        validate_labels(cases, rows, review["capacities"][str(capacity)])
    return cases, review


def score_saved_extraction(cases, review):
    return {str(capacity): score_extraction(cases, _source_rows(capacity),
                                            review["capacities"][str(capacity)])
            for capacity in CAPACITIES}


def _manifest():
    paths = [DATASET, REVIEW, ROOT / "agent_pipeline_v2_1/story_pipeline.py",
             ROOT / "agent_pipeline_v2_1/oracle_benchmark.py",
             ROOT / "agent_pipeline_v2_1/prompts/judge_v2_1.txt",
             ROOT / "agent_pipeline_v2_1/judge.py",
             ROOT / "agent_pipeline_v2_1/retrieval.py",
             ROOT / "agent_pipeline_v2/batch_llm.py",
             ROOT / "agent_pipeline_v3/model.py", ROOT / "agent_pipeline_v2_2/systems.py",
             ROOT / "data/index/CURRENT"]
    index_version = (ROOT / "data/index/CURRENT").read_text(encoding="ascii").strip()
    paths += [ROOT / "data/index/versions" / index_version / name
              for name in ("manifest.json", "metadata.json")]
    paths += [SOURCE / f"capacity_{capacity}" / "extractor_runs.jsonl" for capacity in CAPACITIES]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): _sha(path) for path in paths}


def _ensure_manifest(directory):
    path = directory / "manifest.json"
    identity = {"schema_version": "benchmark-4.1-downstream-v1", "files": _manifest(),
                "retrieval": "hybrid_rrf_no_metadata", "top_k": 5,
                "judge": "v2.1", "judge_batch_size": 8}
    if path.exists():
        if _read(path) != identity:
            raise ValueError("4.1 resume identity mismatch; use a new output directory")
    else:
        atomic_write_json(path, identity)


def _run_one(extraction, llm, retriever):
    from agent_pipeline_v2_1.story_pipeline import (retrieve_story_events,
                                                      judge_story_events, build_story_report)

    started = time.perf_counter()
    retrieval = retrieve_story_events(extraction, retriever, "hybrid", False, 5)
    retrieval_seconds = time.perf_counter() - started
    started = time.perf_counter()
    judge = judge_story_events(retrieval, llm, 8)
    judge_seconds = time.perf_counter() - started
    started = time.perf_counter()
    report = build_story_report(judge)
    report_seconds = time.perf_counter() - started
    return {"result": report, "status": report["status"],
            "stage_seconds": {"retrieval": retrieval_seconds,
                              "judge": judge_seconds, "report": report_seconds}}


def run_downstream(args):
    cases, review = inputs()
    directory = args.output.resolve() if args.output else REPORTS / ("benchmark_4_1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=True)
    _ensure_manifest(directory)
    print("RESULT_DIR=" + str(directory), flush=True)
    remaining = {}
    for capacity in CAPACITIES:
        path = directory / f"CB{capacity}" / "story_runs.jsonl"
        prior = _rows(path) if path.is_file() else []
        if len(prior) != len({row["case_id"] for row in prior}):
            raise ValueError(f"duplicate CB{capacity} story rows")
        remaining[capacity] = (path, prior, {row["case_id"] for row in prior})
    if any(len(prior) < len(cases) for _, prior, _ in remaining.values()):
        # Reuse Benchmark 3 runtime, model, index and retrieval logic unchanged.
        from agent_pipeline_v3.model import ProfiledV2QwenJudge
        from agent_pipeline_v2_1.retrieval import V21Retriever
        from encoder import Encoder
        from retrieval import Retriever
        from agent_pipeline_v3.cli import INDEX

        print("正在加载 Benchmark 3 共享模型和检索器……", flush=True)
        llm = ProfiledV2QwenJudge(args.device)
        dense = Retriever(INDEX, Encoder(device=llm.device, offline=True, precision="float32"))
        lore_metadata = _read(ROOT / "evaluation/agent_pipeline_v2_1_lore_metadata.json")
        retriever = V21Retriever(dense, lore_metadata)
        for capacity in CAPACITIES:
            path, prior, completed = remaining[capacity]
            by_case = {row["case_id"]: row for row in _source_rows(capacity)}
            for case in cases:
                cid = case["id"]
                if cid in completed:
                    continue
                extraction = by_case[cid]["result"]["benchmark_stages"]["extraction"]
                llm.set_story_id(f"CB{capacity}:{cid}")
                print(f"[CB{capacity} {len(prior)+1}/24] {cid}", flush=True)
                result = _run_one(extraction, llm, retriever)
                prior.append({"case_id": cid, "system": "candidate", **result})
                write_jsonl_atomic(path, prior)
                llm.clear_traces()
    return summarize(directory, cases, review)


def summarize(directory, cases=None, review=None):
    if cases is None:
        cases, review = inputs()
    _ensure_manifest(directory)
    extraction = score_saved_extraction(cases, review)
    end_to_end = {}
    downstream_status = {}
    for capacity in CAPACITIES:
        path = directory / f"CB{capacity}" / "story_runs.jsonl"
        if path.is_file() and len(_rows(path)) == len(cases):
            rows = _rows(path)
            end_to_end[str(capacity)] = score_end_to_end(cases, rows, review["capacities"][str(capacity)])
            downstream_status[str(capacity)] = {status: sum(row["status"] == status for row in rows)
                                                for status in ("ok", "partial", "error")}
    result = {"schema_version": "benchmark-4.1-quality-summary-v1",
              "review_provenance": review["provenance"], "source": str(SOURCE),
              "extraction": extraction, "end_to_end_conflict": end_to_end,
              "downstream_status": downstream_status,
              "complete": len(end_to_end) == len(CAPACITIES),
              "notice": "assistant-reviewed草案故事；Hallucination按原文依据而非设定真假评；E2E正类为明确矛盾。"}
    atomic_write_json(directory / "quality_summary.json", result)
    with (directory / "event_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("capacity", "case_id", "event_id", "event",
                                                        "source_text", "label", "gold_ids", "note"))
        writer.writeheader()
        for capacity in CAPACITIES:
            labels = review["capacities"][str(capacity)]
            for row in _source_rows(capacity):
                cid = row["case_id"]
                extraction_row = row["result"]["benchmark_stages"]["extraction"]
                spans = extraction_row["spans"]
                for event in extraction_row["events"]:
                    value = labels[cid][event["id"]]
                    writer.writerow({"capacity": capacity, "case_id": cid, "event_id": event["id"],
                                     "event": event["event"],
                                     "source_text": "".join(spans[sid]["text"] for sid in event["source_ids"]),
                                     "label": value["label"], "gold_ids": ",".join(value["gold_ids"]),
                                     "note": value["note"]})
    lines = ["# Benchmark 4.1：CB2／CB4／CB8 质量复核", "",
             "基于4B冻结提取输出，使用Benchmark 3原下游检索、Judge和Report。人工标签为assistant-reviewed；测试故事与gold仍是草案。", "",
             "| CB | Gold命中 | Extraction Recall | Extraction Precision | Hallucination | Overselect | Duplicate | E2E TP/FP/FN | E2E F1 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for capacity in CAPACITIES:
        key = str(capacity)
        ex = extraction[key]
        e2e = end_to_end.get(key)
        pct = lambda value: "待运行" if value is None else f"{value:.2%}"
        confusion = f"{e2e['tp']}/{e2e['fp']}/{e2e['fn']}" if e2e else "待运行"
        lines.append(f"| {capacity} | {ex['gold_hits']}/{ex['gold_total']} | {pct(ex['recall'])} | {pct(ex['precision'])} | "
                     f"{ex['hallucinated_events']}/{ex['emitted_events']} ({pct(ex['hallucination_rate'])}) | "
                     f"{ex['overselected_events']}/{ex['emitted_events']} ({pct(ex['overselect_rate'])}) | "
                     f"{ex['duplicate_events']} | {confusion} | {pct(e2e['f1'] if e2e else None)} |")
    lines += ["", "定义：Hallucination＝提取事实缺乏引用原文依据；Overselect＝原文支持但不值得设定核对；Duplicate＝已提取事实的重复条目。三类互斥，分母均为输出事件数。", "",
              "E2E F1 以 gold 的明确矛盾为正类；无对应gold的矛盾判断也计FP，漏掉的gold矛盾计FN。", "",
              "## 漏提 Gold", ""]
    for capacity in CAPACITIES:
        lines.append(f"- CB{capacity}: {', '.join(extraction[str(capacity)]['missed_gold']) or '无'}")
    (directory / "quality_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("SUMMARY=" + str(directory / "quality_summary.md"), flush=True)
    return 0 if result["complete"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark 4.1 quality audit")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    extraction = sub.add_parser("score-extraction")
    extraction.add_argument("--output", type=Path)
    run = sub.add_parser("run-downstream")
    run.add_argument("--device", choices=("cuda",), default="cuda")
    run.add_argument("--output", type=Path)
    summary = sub.add_parser("summary")
    summary.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            cases, review = inputs()
            print(json.dumps({"status": "ok", "stories": len(cases), "capacities": CAPACITIES,
                              "review": review["provenance"]}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "score-extraction":
            cases, review = inputs()
            directory = args.output.resolve() if args.output else REPORTS / "extraction_only"
            directory.mkdir(parents=True, exist_ok=True)
            _ensure_manifest(directory)
            summarize(directory, cases, review)
            return 0
        if args.command == "run-downstream":
            return run_downstream(args)
        return summarize(args.result_dir.resolve())
    except (ValueError, OSError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print("错误：" + str(exc), flush=True)
        return 2
