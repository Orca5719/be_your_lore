"""Oracle Judge A/B prompt comparison primitives for Benchmark 2.1A."""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import time

from agent_pipeline_v2.batch_llm import call_json_batch
from agent_pipeline_v2.judge import validate_answer as validate_v1_answer

from .judge import apply_scope_guard, validate_answer, validate_model_answer


ROOT = Path(__file__).resolve().parent.parent
PROMPTS = {
    "v1": ROOT / "agent_pipeline_v2" / "prompts" / "judge_v1.txt",
    "v2.1": ROOT / "agent_pipeline_v2_1" / "prompts" / "judge_v2_1.txt",
}
LABELS = ("consistent", "contradiction", "uncertain")


def build_messages(item: dict, prompt_version: str) -> list[dict]:
    if prompt_version not in PROMPTS:
        raise ValueError("prompt_version必须为v1或v2.1")
    fact = item["fact"]
    lore = [
        {"evidence_id": f"L{number}", "text": chunk["text"], "heading_path": chunk.get("heading_path", [])}
        for number, chunk in enumerate(item["lore"], 1)
    ]
    payload = {
        "event": {
            "actors": [fact["subject"]],
            "event": fact["normalized_fact"],
            "mental_state": None,
            "explicit": True,
            "modality": "observed",
            "conditions": [],
            "check_reason": fact["dimension"],
        },
        "story_context": item["story_context"],
        "lore": lore,
    }
    return [
        {"role": "system", "content": PROMPTS[prompt_version].read_text(encoding="utf-8")},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def finalize_answer(value: dict, evidence: list[dict], prompt_version: str) -> dict:
    if prompt_version == "v1":
        validate_v1_answer(value, evidence)
        return copy.deepcopy(value)
    if prompt_version != "v2.1":
        raise ValueError("prompt_version必须为v1或v2.1")
    validate_model_answer(value, evidence)
    result = apply_scope_guard(value)
    validate_answer({key: result[key] for key in ("verdict", "citations", "reason", "assessment")}, evidence)
    return result


def _class_metrics(rows: list[dict], label: str) -> dict:
    tp = sum(row.get("status") == "ok" and row["expected_verdict"] == label and row.get("predicted_verdict") == label for row in rows)
    fp = sum(row.get("status") == "ok" and row["expected_verdict"] != label and row.get("predicted_verdict") == label for row in rows)
    fn = sum(row["expected_verdict"] == label and (row.get("status") != "ok" or row.get("predicted_verdict") != label) for row in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def score_oracle_rows(rows: list[dict]) -> dict:
    confusion = defaultdict(Counter)
    for row in rows:
        predicted = row.get("predicted_verdict") if row.get("status") == "ok" else "error"
        confusion[row["expected_verdict"]][predicted] += 1
    per_class = {label: _class_metrics(rows, label) for label in LABELS}
    f1_values = [metrics["f1"] for metrics in per_class.values() if metrics["f1"] is not None]
    correct = sum(row.get("status") == "ok" and row.get("predicted_verdict") == row["expected_verdict"] for row in rows)
    return {
        "facts_total": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "per_class": per_class,
        "confusion": {expected: dict(values) for expected, values in confusion.items()},
        "contradiction_false_positive": per_class["contradiction"]["fp"],
        "contradiction_false_negative": per_class["contradiction"]["fn"],
        "unsupported_inference_count": sum(bool(row.get("unsupported_inference")) for row in rows),
        "parse_failure_count": sum(row.get("status") != "ok" for row in rows),
    }


def run_oracle_items(items: list[dict], llm, prompt_version: str, batch_size: int = 8, progress=None) -> dict:
    if prompt_version not in PROMPTS:
        raise ValueError("prompt_version必须为v1或v2.1")
    started = time.perf_counter()
    rows = []
    batch_reports = []
    for offset in range(0, len(items), batch_size):
        group = items[offset : offset + batch_size]
        if progress:
            progress({"completed": offset, "total": len(items), "batch_size": len(group)})
        requests = [{"request_id": item["fixture_id"], "messages": build_messages(item, prompt_version)} for item in group]
        if prompt_version == "v1":
            validators = [(lambda lore: lambda value: validate_v1_answer(value, lore))(item["lore"]) for item in group]
        else:
            validators = [(lambda lore: lambda value: validate_model_answer(value, lore))(item["lore"]) for item in group]
        values, report = call_json_batch(llm, requests, max_output=768, validators=validators)
        batch_reports.append(report)
        transports = {row["request_id"]: row for row in report["rows"]}
        for item, value in zip(group, values):
            transport = transports[item["fixture_id"]]
            row = {
                "fixture_id": item["fixture_id"],
                "case_id": item["case_id"],
                "gold_fact_id": item["gold_fact_id"],
                "expected_verdict": item["expected_verdict"],
                "status": transport["status"],
                "attempts": transport["attempts"],
                "predicted_verdict": None,
                "transport": transport,
            }
            if value is not None:
                final = finalize_answer(value, item["lore"], prompt_version)
                row["predicted_verdict"] = final["verdict"]
                row["answer"] = final
                assessment = value.get("assessment", {})
                row["unsupported_inference"] = value.get("verdict") == "contradiction" and (
                    assessment.get("same_subject") is not True
                    or assessment.get("evidence_applicable") is not True
                    or assessment.get("relation") != "direct_conflict"
                    or bool(assessment.get("assumptions"))
                    or assessment.get("can_both_be_true") is True
                )
            else:
                row["error"] = transport.get("error", "模型未返回有效结果")
            rows.append(row)
    metrics = score_oracle_rows(rows)
    metrics["total_judge_seconds"] = time.perf_counter() - started
    metrics["facts_per_second"] = len(items) / metrics["total_judge_seconds"] if metrics["total_judge_seconds"] else None
    return {
        "schema_version": "agent-pipeline-v2.1-oracle-run-v1",
        "prompt_version": prompt_version,
        "batch_size": batch_size,
        "status": "ok" if metrics["parse_failure_count"] == 0 else "partial",
        "items": rows,
        "metrics": metrics,
        "batch_reports": batch_reports,
    }


def build_comparison_report(results: list[dict], fixture_sha256: str) -> dict:
    by_prompt = {result["prompt_version"]: result for result in results}
    if set(by_prompt) != {"v1", "v2.1"}:
        raise ValueError("Oracle比较必须同时包含v1与v2.1")
    old = by_prompt["v1"].get("metrics", {})
    new = by_prompt["v2.1"].get("metrics", {})

    def delta(key):
        if not isinstance(old.get(key), (int, float)) or not isinstance(new.get(key), (int, float)):
            return None
        return round(new[key] - old[key], 12)

    return {
        "schema_version": "agent-pipeline-v2.1-oracle-comparison-v1",
        "status": "ok" if all(result.get("status") == "ok" for result in results) else "partial",
        "fixture_sha256": fixture_sha256,
        "evaluation": "Oracle Retrieval; same 72 facts and lore for both prompts",
        "results": by_prompt,
        "comparison": {
            "accuracy_delta": delta("accuracy"),
            "macro_f1_delta": delta("macro_f1"),
            "contradiction_false_positive_delta": delta("contradiction_false_positive"),
            "contradiction_false_negative_delta": delta("contradiction_false_negative"),
            "unsupported_inference_delta": delta("unsupported_inference_count"),
        },
    }


def _fmt(value) -> str:
    return "N/A" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def render_comparison_markdown(report: dict) -> str:
    results = report["results"]
    lines = [
        "# Benchmark 2.1A — Oracle Judge",
        "",
        "同一批72条事实使用 Oracle Retrieval，比较冻结的 Judge v1 与 Judge v2.1。",
        "",
        "| Metric | Judge v1 | Judge v2.1 | Delta |",
        "|---|---:|---:|---:|",
    ]
    fields = [
        ("Accuracy", "accuracy", "accuracy_delta"),
        ("Macro-F1", "macro_f1", "macro_f1_delta"),
        ("Contradiction FP", "contradiction_false_positive", "contradiction_false_positive_delta"),
        ("Contradiction FN", "contradiction_false_negative", "contradiction_false_negative_delta"),
        ("Unsupported inference", "unsupported_inference_count", "unsupported_inference_delta"),
    ]
    for label, key, delta_key in fields:
        lines.append(f"| {label} | {_fmt(results['v1'].get('metrics', {}).get(key))} | {_fmt(results['v2.1'].get('metrics', {}).get(key))} | {_fmt(report['comparison'].get(delta_key))} |")
    lines.extend(["", f"Status: {report['status']}", ""])
    return "\n".join(lines)
