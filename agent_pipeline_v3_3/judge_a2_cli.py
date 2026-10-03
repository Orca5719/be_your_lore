"""Extended 3.3A Judge ablations, frozen to Benchmark 3 retrieval."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from agent_pipeline_v2_2.scoring import score_system
from agent_pipeline_v3.metrics import aggregate_profile
from agent_pipeline_v3.model import MODEL, REVISION, ProfiledV2QwenJudge
from agent_pipeline_v3_2.cli import _candidate_review
from agent_pipeline_v3_2.frozen import load_and_validate_source, sha256
from agent_pipeline_v3_2.runner import write_json_atomic, write_jsonl_atomic

from .judge_a2 import VARIANTS, judge_frozen_retrieval
from .judge_a_cli import DATASET, DEFAULT_SOURCE, LEAN_RESULT, REPORT_ROOT, REVIEW, ROOT
from .judge_a_report import classification_metrics, verdict_changes
from .judge_a_runner import build_candidate_row


A1_RESULT = REPORT_ROOT / "benchmark_3_3A_20261003T154610Z_7305ef" / "judge_a_summary.json"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_a1_result(path: Path) -> Path:
    if path.suffix.lower() == ".md":
        return path.with_suffix(".json").resolve()
    if path.suffix.lower() == ".json":
        return path.resolve()
    if path.suffix:
        raise ValueError("--a1-result须为结果目录、Markdown或JSON报告")
    return (path / "judge_a_summary.json").resolve()


def build_manifest(source_hashes: dict[str, str], device: str) -> dict:
    if device != "cuda":
        raise ValueError("3.3A追加实验固定CUDA")
    files = ("judge_a2.py", "judge_a2_cli.py", "judge_a.py", "judge_a_runner.py", "judge_a_report.py")
    return {
        "schema_version": "agent-pipeline-v3.3a2-manifest-v1",
        "source_hashes": dict(source_hashes), "model": MODEL, "revision": REVISION,
        "device": device, "batch_size": 8,
        "prompt_hashes": {variant: sha256(ROOT / "agent_pipeline_v3_3" / "prompts" / f"judge_a2_{variant}.txt")
                          for variant in VARIANTS},
        "implementation_hashes": {name: sha256(ROOT / "agent_pipeline_v3_3" / name) for name in files},
        "full_judge_input_builder_sha256": sha256(ROOT / "agent_pipeline_v2_1" / "oracle_benchmark.py"),
    }


def _run_variant(source_rows: list[dict], llm, directory: Path, variant: str) -> list[dict]:
    path = directory / variant / "judge_runs.jsonl"
    existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    source_ids = [row["case_id"] for row in source_rows]
    if [row["case_id"] for row in existing] != source_ids[:len(existing)]:
        raise ValueError("续跑案例顺序或ID与冻结输入不符")
    llm.collector.resume_after([call for row in existing for call in row.get("calls", [])])
    for source_row in source_rows[len(existing):]:
        llm.set_story_id(source_row["case_id"])
        before = len(llm.collector.records)
        started = time.perf_counter()
        judge = judge_frozen_retrieval(source_row["result"]["benchmark_stages"]["retrieval"], llm, variant, 8)
        elapsed = time.perf_counter() - started
        row = build_candidate_row(source_row, judge, llm.collector.records[before:], elapsed, variant)
        existing.append(row)
        write_jsonl_atomic(path, existing)
        print(f"  [{variant} {len(existing)}/{len(source_rows)}] {source_row['case_id']}", flush=True)
    return existing


def _row(quality: dict, perf: dict, complete: int = 24) -> dict:
    return {"quality": quality, "classification": classification_metrics(quality["judge"]["confusion"]),
            "performance": perf, "complete_stories": complete}


def _format(value, kind):
    return "-" if value is None else f"{value * 100:.2f}%" if kind == "pct" else f"{value:.3f}" if kind == "float" else str(value)


def _write_report(directory: Path, summary: dict, changes: list[dict], calls: list[dict]) -> Path:
    write_json_atomic(directory / "judge_a2_summary.json", summary)
    with (directory / "verdict_changes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ("variant", "case_id", "event_id", "old_verdict", "new_verdict", "changed", "evidence_ids", "guard_reason")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(changes)
    with (directory / "judge_calls.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ("variant", "call_id", "story_id", "status", "batch_size", "input_tokens", "output_tokens",
                  "prefill_time", "decode_time", "total_time", "ttft_ms", "peak_allocated", "peak_reserved", "is_retry")
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(calls)
    columns = [("Full B3", summary["baseline"]), ("Verdict Only", summary["lean"]),
               ("Structured", summary["a1"]["structured"]), ("Short 80", summary["a1"]["short-reason"])]
    columns += [(name, summary["variants"][name]) for name in VARIANTS]
    lines = ["# Benchmark 3.3A — additional output ablations", "",
             "冻结Benchmark 3的80个事实和检索证据，仅重跑右侧三种Judge。既有四列来自历史实测，不重新运行。",
             "", "| Metric | " + " | ".join(name for name, _ in columns) + " |",
             "|---|" + "---:|" * len(columns)]
    measures = (
        ("Judge Accuracy", lambda x: x["quality"]["judge"]["accuracy"], "pct"),
        ("Macro-F1", lambda x: x["classification"]["macro_f1"], "pct"),
        ("Uncertain Recall", lambda x: x["classification"]["uncertain_recall"], "pct"),
        ("Judge FP", lambda x: x["quality"]["judge"]["fp"], "int"),
        ("Judge FN", lambda x: x["quality"]["judge"]["fn_total"], "int"),
        ("End-to-End F1", lambda x: x["quality"]["end_to_end_conflict"]["f1"], "pct"),
        ("Calls", lambda x: x["performance"]["calls"], "int"),
        ("Retry calls", lambda x: x["performance"]["retry_calls"], "int"),
        ("Output tokens", lambda x: x["performance"]["output_tokens"], "int"),
        ("Prefill s", lambda x: x["performance"]["prefill_seconds"], "float"),
        ("Decode s", lambda x: x["performance"]["decode_seconds"], "float"),
        ("Judge total s", lambda x: x["performance"]["total_seconds"], "float"),
    )
    for name, getter, kind in measures:
        lines.append("| " + name + " | " + " | ".join(_format(getter(row), kind) for _, row in columns) + " |")
    lines += ["", "## Verdict movement from Full B3", ""]
    for variant in VARIANTS:
        rows = [row for row in changes if row["variant"] == variant]
        lines.append(f"- {variant}: changed {sum(row['changed'] for row in rows)}/{len(rows)}; "
                     f"old uncertain → decisive {sum(row['old_verdict'] == 'uncertain' and row['new_verdict'] != 'uncertain' for row in rows)}; "
                     f"guarded downgrades {sum(bool(row['guard_reason']) for row in rows)}.")
    lines += ["", "Macro-F1与Uncertain Recall在有完整提取及证据的gold事实中计算。",
              "本轮为固定开发集单次实验；速度与质量须联合判断，不自动选新默认Judge。", ""]
    path = directory / "judge_a2_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def summarize(directory: Path, source: dict, a1_result: Path = A1_RESULT) -> dict:
    dataset = _read(DATASET)
    review = _candidate_review(_read(REVIEW))
    a1 = _read(a1_result)
    if Path(a1["source"]).resolve() != source["source"]:
        raise ValueError("3.3A原报告与冻结输入不匹配")
    lean = _read(LEAN_RESULT)
    baseline_quality = _read(source["source"] / "quality_reference.json")["official_metrics"]
    baseline_perf = _read(source["source"] / "profile_summary.json")["components"]["judge"]
    variants = {}
    changes = []
    calls = []
    for variant in VARIANTS:
        path = directory / variant / "judge_runs.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 24:
            raise ValueError(f"{variant}: 仅完成{len(rows)}/24篇")
        variant_calls = [call for row in rows for call in row.get("calls", [])]
        perf = aggregate_profile(variant_calls, rows, 0.0)["components"]["judge"]
        quality = score_system(dataset["cases"], rows, review, 5)
        variants[variant] = _row(quality, perf, sum(row["status"] == "ok" for row in rows))
        changes.extend(verdict_changes(source["rows"], rows, variant))
        calls.extend({"variant": variant, **call} for call in variant_calls)
    summary = {"schema_version": "agent-pipeline-v3.3a2-summary-v1", "source": str(source["source"]),
               "stories": 24, "facts": 80,
               "baseline": _row(baseline_quality, baseline_perf),
               "lean": _row(lean["quality"], lean["performance"]["components"]["judge"]),
               "a1": {name: {key: a1["variants"][name][key] for key in ("quality", "classification", "performance")}
                      for name in ("structured", "short-reason")},
               "variants": variants}
    _write_report(directory, summary, changes, calls)
    return summary


def run_command(args) -> int:
    source = load_and_validate_source(args.source_result)
    if args.device != "cuda":
        raise ValueError("3.3A追加实验固定CUDA")
    a1_result = resolve_a1_result(args.a1_result)
    if Path(_read(a1_result)["source"]).resolve() != source["source"]:
        raise ValueError("3.3A原报告与冻结输入不匹配")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = (args.output or REPORT_ROOT / f"benchmark_3_3A2_{stamp}_{uuid.uuid4().hex[:6]}").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(source["hashes"], args.device)
    manifest["source"] = str(source["source"])
    manifest["inputs"] = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                          for path in (DATASET, REVIEW, LEAN_RESULT)}
    manifest["a1_result"] = str(a1_result)
    manifest["a1_result_sha256"] = sha256(a1_result)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and _read(manifest_path) != manifest:
        raise ValueError("续跑manifest不匹配；不得混用旧结果")
    write_json_atomic(manifest_path, manifest)
    print("RESULT_DIR=" + str(directory), flush=True)
    remaining = []
    for variant in VARIANTS:
        path = directory / variant / "judge_runs.jsonl"
        if not path.exists() or len(path.read_text(encoding="utf-8").splitlines()) < 24:
            remaining.append(variant)
    if remaining:
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
            _run_variant(source["rows"], llm, directory, variant)
    summary = summarize(directory, source, a1_result)
    print("SUMMARY=" + str(directory / "judge_a2_summary.md"), flush=True)
    return 0 if all(row["complete_stories"] == 24 for row in summary["variants"].values()) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark 3.3A additional Judge ablations")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE)
    run.add_argument("--a1-result", type=Path, default=A1_RESULT)
    run.add_argument("--device", choices=("cuda",), default="cuda")
    run.add_argument("--output", type=Path)
    run.set_defaults(handler=run_command)
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
