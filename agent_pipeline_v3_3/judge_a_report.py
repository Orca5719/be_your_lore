"""Offline four-way Judge A/B comparison, including uncertain recall."""

from __future__ import annotations

import csv
import json
from pathlib import Path


LABELS = ("consistent", "contradiction", "uncertain")


def classification_metrics(confusion: dict) -> dict:
    per_class = {}
    for label in LABELS:
        tp = confusion.get(label, {}).get(label, 0)
        actual = sum(confusion.get(label, {}).values())
        predicted = sum(confusion.get(gold, {}).get(label, 0) for gold in LABELS)
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": actual, "precision": precision, "recall": recall, "f1": f1}
    return {"macro_f1": sum(value["f1"] for value in per_class.values()) / len(LABELS),
            "uncertain_recall": per_class["uncertain"]["recall"], "per_class": per_class}


def verdict_changes(source_rows: list[dict], candidate_rows: list[dict], variant: str) -> list[dict]:
    old = {(row["case_id"], item["event_id"]): item
           for row in source_rows for item in row["result"]["benchmark_stages"]["judge"]["items"]}
    changes = []
    for row in candidate_rows:
        for item in row["result"]["judge"]["items"]:
            previous = old[row["case_id"], item["event_id"]]
            changes.append({"variant": variant, "case_id": row["case_id"], "event_id": item["event_id"],
                            "old_verdict": previous["verdict"], "new_verdict": item["verdict"],
                            "changed": previous["verdict"] != item["verdict"],
                            "evidence_ids": "|".join(item.get("evidence_ids", [])),
                            "guard_reason": item.get("guard_reason", "")})
    return changes


def write_report(directory: Path, summary: dict, changes: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "judge_a_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (directory / "verdict_changes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("variant", "case_id", "event_id", "old_verdict",
                                                         "new_verdict", "changed", "evidence_ids", "guard_reason"))
        writer.writeheader()
        writer.writerows(changes)
    with (directory / "judge_calls.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ("variant", "call_id", "story_id", "status", "batch_size", "input_tokens", "output_tokens",
                  "prefill_time", "decode_time", "total_time", "ttft_ms", "peak_allocated", "peak_reserved", "is_retry")
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for variant in ("structured", "short-reason"):
            for call in summary["variants"][variant]["calls"]:
                writer.writerow({"variant": variant, **call})
    lines = ["# Benchmark 3.3A — Compact Reasoning Judge", "",
             "冻结 Benchmark 3 的80个事实与检索证据。两种候选各跑一轮Judge；Extractor和Retrieval未重跑。",
             "", "| Metric | Full Judge (B3) | Verdict Only (3.2A) | Structured | Short Reason |",
             "|---|---:|---:|---:|---:|"]
    columns = [summary["baseline"], summary["lean"], summary["variants"]["structured"],
               summary["variants"]["short-reason"]]
    metrics = (
        ("Judge Accuracy", lambda x: x["quality"]["judge"]["accuracy"], "pct"),
        ("Judge Macro-F1", lambda x: x["classification"]["macro_f1"], "pct"),
        ("Uncertain Recall", lambda x: x["classification"]["uncertain_recall"], "pct"),
        ("Judge FP", lambda x: x["quality"]["judge"]["fp"], "int"),
        ("Judge FN", lambda x: x["quality"]["judge"]["fn_total"], "int"),
        ("End-to-End F1", lambda x: x["quality"]["end_to_end_conflict"]["f1"], "pct"),
        ("Calls", lambda x: x["performance"]["calls"], "int"),
        ("Retry calls", lambda x: x["performance"]["retry_calls"], "int"),
        ("Output tokens", lambda x: x["performance"]["output_tokens"], "int"),
        ("Prefill s", lambda x: x["performance"]["prefill_seconds"], "sec"),
        ("Decode s", lambda x: x["performance"]["decode_seconds"], "sec"),
        ("Judge total s", lambda x: x["performance"]["total_seconds"], "sec"),
        ("TTFT p50 ms", lambda x: x["performance"]["ttft_p50_ms"], "sec"),
        ("Peak allocated GiB", lambda x: None if x["performance"]["peak_allocated"] is None
         else x["performance"]["peak_allocated"] / 1024**3, "sec"),
    )
    for label, getter, fmt in metrics:
        values = []
        for column in columns:
            value = getter(column)
            values.append("-" if value is None else f"{value * 100:.2f}%" if fmt == "pct" else str(value) if fmt == "int" else f"{value:.3f}")
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    lines += ["", "## Verdict movement from Benchmark 3", ""]
    for variant in ("structured", "short-reason"):
        rows = [row for row in changes if row["variant"] == variant]
        decisive = sum(row["old_verdict"] == "uncertain" and row["new_verdict"] != "uncertain" for row in rows)
        lines.append(f"- {variant}: {sum(row['changed'] for row in rows)}/{len(rows)} changed; "
                     f"old uncertain → decisive: {decisive}; guarded downgrades: "
                     f"{sum(bool(row['guard_reason']) for row in rows)}.")
    lines += ["", "这是固定开发集上的实验结果；不以端到端F1单独判定新Judge可用。",
              "证据原文由evidence ID映射，不要求模型复制。Short Reason仅有一句限长理由。", ""]
    path = directory / "judge_a_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
