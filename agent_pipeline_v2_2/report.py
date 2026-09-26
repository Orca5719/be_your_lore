from __future__ import annotations

import json
from pathlib import Path


METRICS = ("extraction", "retrieval", "judge", "unsupported_reasoning", "end_to_end_conflict")


def _delta(before, after):
    if before is None or after is None:
        return {"absolute": None, "relative": None}
    absolute = after - before
    return {"absolute": absolute, "relative": absolute / before if before else None}


def build_summary(baseline: dict, candidate: dict, *, performance: dict, attribution: dict, manifest: dict, case_deltas: list[dict] | None = None) -> dict:
    comparisons = {
        "extraction_recall": _delta(baseline["extraction"]["recall"], candidate["extraction"]["recall"]),
        "retrieval_recall_at_5": _delta(baseline["retrieval"]["recall_at_5"], candidate["retrieval"]["recall_at_5"]),
        "judge_accuracy": _delta(baseline["judge"]["accuracy"], candidate["judge"]["accuracy"]),
        "end_to_end_precision": _delta(baseline["end_to_end_conflict"]["precision"], candidate["end_to_end_conflict"]["precision"]),
        "end_to_end_recall": _delta(baseline["end_to_end_conflict"]["recall"], candidate["end_to_end_conflict"]["recall"]),
        "end_to_end_f1": _delta(baseline["end_to_end_conflict"]["f1"], candidate["end_to_end_conflict"]["f1"]),
    }
    return {
        "schema_version": "agent-pipeline-benchmark-3-summary-v1", "status": "ok",
        "configuration": manifest.get("identity", {}), "baseline": baseline, "candidate": candidate,
        "deltas": comparisons, "performance": performance, "first_failure_attribution": attribution,
        "case_deltas": case_deltas or [],
        "reconciliation": {
            "quality": "quality.json", "attribution": "attribution.json",
            "end_to_end_rows": "end_to_end.jsonl", "paired_rows": "paired.jsonl", "review": "review.json",
        },
        "freeze_ready": False,
        "limitations": [
            "24篇故事与测试设定不能代表真实长篇分布。",
            "人工审核账本决定语义指标上限，assistant-reviewed不等同于作者确认。",
            "模型能力限制与结构/执行失败分别报告，不能由单次总分区分。",
        ],
    }


def render_markdown(summary: dict) -> str:
    b, c = summary["baseline"], summary["candidate"]
    def pct(value): return "N/A" if value is None else f"{value:.2%}"
    rows = [
        ("Extraction Recall", b["extraction"]["recall"], c["extraction"]["recall"]),
        ("Retrieval Recall@5", b["retrieval"]["recall_at_5"], c["retrieval"]["recall_at_5"]),
        ("Judge Accuracy", b["judge"]["accuracy"], c["judge"]["accuracy"]),
        ("Unsupported Reasoning", b["unsupported_reasoning"]["rate"], c["unsupported_reasoning"]["rate"]),
        ("End-to-End Precision", b["end_to_end_conflict"]["precision"], c["end_to_end_conflict"]["precision"]),
        ("End-to-End Recall", b["end_to_end_conflict"]["recall"], c["end_to_end_conflict"]["recall"]),
        ("End-to-End F1", b["end_to_end_conflict"]["f1"], c["end_to_end_conflict"]["f1"]),
    ]
    lines = ["# Agent Pipeline Benchmark 2.2", "", "## Configuration", "", "```json", json.dumps(summary["configuration"], ensure_ascii=False, indent=2), "```", "", "## Quality", "", "| Metric | Baseline | Candidate |", "|---|---:|---:|"]
    lines += [f"| {name} | {pct(left)} | {pct(right)} |" for name, left, right in rows]
    lines += ["", "## Judge confusion matrices", "", "```json", json.dumps({"baseline": b["judge"].get("confusion", {}), "candidate": c["judge"].get("confusion", {})}, ensure_ascii=False, indent=2), "```", "", "## Error attribution", "", "```json", json.dumps(summary["first_failure_attribution"], ensure_ascii=False, indent=2), "```", "", "## Performance", "", "```json", json.dumps(summary["performance"], ensure_ascii=False, indent=2), "```", "", "## Reconciliation", "", "```json", json.dumps(summary["reconciliation"], ensure_ascii=False, indent=2), "```", "", "## Limitations", ""]
    lines += [f"- {item}" for item in summary["limitations"]]
    return "\n".join(lines) + "\n"


def write_summary(directory: Path, summary: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "benchmark_2_2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "benchmark_2_2_summary.md").write_text(render_markdown(summary), encoding="utf-8")
