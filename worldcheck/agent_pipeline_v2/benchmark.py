"""Quality and throughput primitives for the Agent Pipeline v2 benchmark."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import time

from .batch_llm import call_json_batch
from .judge import validate_answer


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _prf(tp, fp, fn):
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def score_story_quality(cases: list[dict], run_rows: list[dict], review: dict, k: int = 5) -> dict:
    # v2 review ledgers keep the final mapping under event_mapping; accept the
    # old plain mapping as well so historical reports remain readable.
    if "event_mapping" in review:
        review_mapping = review.get("event_mapping", {})
        event_labels_by_case = review.get("system_event_labels", {})
        finding_labels_by_case = review.get("system_finding_labels", {})
    else:
        review_mapping = review
        event_labels_by_case = {}
        finding_labels_by_case = {}
    by_case = {row["case_id"]: row for row in run_rows}
    extraction_hits = extraction_total = retrieval_hits = retrieval_total = 0
    judge_correct = judge_eligible = 0
    tp = fp = fn = 0
    confusion = defaultdict(Counter)
    label_codes = {"一致": "consistent", "矛盾": "contradiction", "不确定": "uncertain"}
    for case in cases:
        result = by_case.get(case["id"], {}).get("result", {})
        judge = result.get("judge", {})
        events = {event["id"]: event for event in judge.get("events", [])}
        judge_items = {item["event_id"]: item for item in judge.get("items", [])}
        retrieval_items = {item["event_id"]: item for item in judge.get("retrieval", {}).get("items", [])}
        mapping = review_mapping.get(case["id"], {})
        valid_events = {event_id for event_id, label in event_labels_by_case.get(case["id"], {}).items() if label.get("label") == "valid_checkable"}
        conflict_gold = {gold["id"] for gold in case.get("gold_facts", []) if gold["expected_verdict"] == "矛盾"}
        covered_conflicts = set()
        positive_predictions = {event_id for event_id, item in judge_items.items() if item.get("status") == "ok" and item.get("verdict") == "contradiction"}
        for gold in case.get("gold_facts", []):
            mapped = [event_id for event_id in mapping.get(gold["id"], []) if event_id in events and (not valid_events or event_id in valid_events)]
            extraction_total += 1
            extraction_hits += bool(mapped)
            alternatives = gold.get("acceptable_evidence_sets", []) or gold.get("minimum_evidence_sets", [])
            retrieved_ok = False
            if alternatives:
                retrieval_total += 1
                for event_id in mapped:
                    item = retrieval_items.get(event_id, {})
                    found = {chunk["id"] for chunk in item.get("evidence", [])[:k]} if item.get("status") == "ok" else set()
                    if any(set(group) <= found for group in alternatives):
                        retrieved_ok = True
                        break
                retrieval_hits += retrieved_ok
            retrieval_ready = any(retrieval_items.get(event_id, {}).get("status") == "ok" for event_id in mapped)
            eligible = bool(mapped) and retrieval_ready and (not alternatives or retrieved_ok)
            if eligible:
                judge_eligible += 1
                predictions = {judge_items[event_id].get("verdict") for event_id in mapped if judge_items.get(event_id, {}).get("status") == "ok"}
                expected = label_codes[gold["expected_verdict"]]
                predicted = next(iter(predictions)) if len(predictions) == 1 else "mixed_or_missing"
                judge_correct += expected in predictions
                confusion[expected][predicted] += 1
            if gold["id"] in conflict_gold and any(event_id in positive_predictions for event_id in mapped):
                covered_conflicts.add(gold["id"])
        reverse = {event_id: {gold_id for gold_id, event_ids in mapping.items() if event_id in event_ids} for event_id in positive_predictions}
        tp += len(covered_conflicts)
        fn += len(conflict_gold - covered_conflicts)
        fp += sum(not bool(gold_ids & conflict_gold) for gold_ids in reverse.values())
    extraction = {"hits": extraction_hits, "gold_total": extraction_total, "recall": _ratio(extraction_hits, extraction_total)}
    retrieval = {"hits": retrieval_hits, "gold_total": retrieval_total, "k": k, "recall": _ratio(retrieval_hits, retrieval_total)}
    judge_metrics = {"correct": judge_correct, "eligible": judge_eligible, "accuracy": _ratio(judge_correct, judge_eligible), "confusion": {key: dict(value) for key, value in confusion.items()}}
    detection = _prf(tp, fp, fn)
    emitted_events = sum(len(labels) for labels in event_labels_by_case.values())
    hallucinated_events = sum(1 for labels in event_labels_by_case.values() for label in labels.values() if label.get("label") == "hallucinated")
    return {
        "recall_extraction": extraction,
        "recall_retrieval_at_k": retrieval,
        "accuracy_judge": judge_metrics,
        "end_to_end_conflict": detection,
        "extraction": {"recall": {"value": extraction["recall"], "hits": extraction_hits, "gold_total": extraction_total}, "hallucination_rate": {"value": _ratio(hallucinated_events, emitted_events), "count": hallucinated_events, "total": emitted_events}},
        "retrieval": {"recall_at_k": {"value": retrieval["recall"], "k": k, "hits": retrieval_hits, "gold_total": retrieval_total}},
        "judge": {"pipeline": {"accuracy": {"value": judge_metrics["accuracy"], "correct": judge_correct, "eligible": judge_eligible}}},
        "detection": {"conflict": detection},
        "scope": "24-story benchmark with assistant-reviewed gold-to-event and finding mappings.",
    }


def _fixture_messages(item: dict, prompt: str) -> list[dict]:
    lore = [
        {"evidence_id": f"L{number}", "text": chunk["text"], "heading_path": chunk.get("heading_path", [])}
        for number, chunk in enumerate(item["lore"], 1)
    ]
    payload = {
        "event": {"actors": item.get("actors", []), "event": item["fact"], "modality": "observed", "conditions": [], "check_reason": "benchmark_claim"},
        "story_context": item["story_context"],
        "lore": lore,
    }
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def score_judge_results(results: list[dict]) -> dict:
    from .scoring.judge import classification_metrics

    total = len(results)
    correct = sum(row.get("status") == "ok" and row.get("predicted_verdict") == row.get("expected_verdict") for row in results)
    failures = sum(row.get("status") != "ok" for row in results)
    retry_count = sum(row.get("attempts", 0) > 1 for row in results)
    per_class = {}
    confusion = defaultdict(Counter)
    for row in results:
        expected = row["expected_verdict"]
        predicted = row.get("predicted_verdict") if row.get("status") == "ok" else "error"
        confusion[expected][predicted] += 1
    for expected, counts in confusion.items():
        class_total = sum(counts.values())
        per_class[expected] = {"correct": counts.get(expected, 0), "total": class_total, "accuracy": counts.get(expected, 0) / class_total if class_total else None}
    expanded_rows = [{
        "expected": {"consistent": "一致", "contradiction": "矛盾", "uncertain": "不确定"}.get(row["expected_verdict"], row["expected_verdict"]),
        "predicted": {"consistent": "一致", "contradiction": "矛盾", "uncertain": "不确定"}.get(row.get("predicted_verdict"), None) if row.get("status") == "ok" else None,
    } for row in results]
    classification = classification_metrics(expanded_rows)
    return {
        "correct": correct,
        "facts_total": total,
        "judge_accuracy": correct / total if total else None,
        "per_class_accuracy": per_class,
        "confusion": {expected: dict(counts) for expected, counts in confusion.items()},
        "parse_failure_count": failures,
        "parse_failure_rate": failures / total if total else None,
        "retry_fact_count": retry_count,
        "retry_rate": retry_count / total if total else None,
        "macro_f1": classification["macro_f1"],
        "per_class": classification["per_class"],
        "classification": classification,
    }


def _generation_totals(batch_reports: list[dict]) -> dict:
    calls = [call for report in batch_reports for call in report.get("batch_calls", [])]
    timings = [call.get("timing", {}) for call in calls]
    useful = sum(int(timing.get("useful_input_tokens", sum(timing.get("input_tokens", []))) or 0) for timing in timings)
    padded = sum(int(timing.get("padded_input_tokens", 0) or 0) for timing in timings)
    return {
        "main_batch_count": sum(call.get("attempt") == 1 for call in calls),
        "retry_batch_count": sum(call.get("attempt") == 2 for call in calls),
        "generation_seconds": sum(float(timing.get("seconds", 0) or 0) for timing in timings),
        "useful_input_tokens": useful,
        "padded_input_tokens": padded,
        "padding_ratio": padded / (useful + padded) if useful + padded else 0.0,
        "generated_tokens": sum(int(timing.get("total_generated_tokens", sum(timing.get("generated_tokens", []))) or 0) for timing in timings),
    }


def judge_fixture_items(items: list[dict], llm, batch_size: int, progress=None) -> dict:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size必须为正整数")
    if not isinstance(items, list) or not items:
        raise ValueError("Judge fixture items不能为空")
    prompt = (Path(__file__).parent / "prompts" / "judge_v1.txt").read_text(encoding="utf-8")
    started = time.perf_counter()
    results = []
    batch_reports = []
    oom = False
    for offset in range(0, len(items), batch_size):
        group = items[offset : offset + batch_size]
        if progress:
            progress({"batch_number": len(batch_reports) + 1, "batch_size": len(group), "completed": offset, "total": len(items)})
        requests = [{"request_id": item["fixture_id"], "messages": _fixture_messages(item, prompt)} for item in group]
        validators = [(lambda lore: lambda value: validate_answer(value, lore))(item["lore"]) for item in group]
        values, call_report = call_json_batch(llm, requests, max_output=768, validators=validators)
        batch_reports.append(call_report)
        rows = {row["request_id"]: row for row in call_report["rows"]}
        for item, value in zip(group, values):
            transport = rows[item["fixture_id"]]
            result = {
                "fixture_id": item["fixture_id"],
                "expected_verdict": item["expected_verdict_code"],
                "status": transport["status"],
                "attempts": transport["attempts"],
                "predicted_verdict": None,
                "transport": transport,
            }
            if value is None:
                result["error"] = transport["error"]
            else:
                verdict = value["verdict"]
                assessment = value["assessment"]
                relation = "direct_support" if verdict == "consistent" else "direct_conflict"
                if verdict != "uncertain" and (
                    assessment["same_subject"] is not True
                    or assessment["evidence_applicable"] is not True
                    or assessment["assumptions"]
                    or assessment["relation"] != relation
                ):
                    verdict = "uncertain"
                    result["scope_guard_applied"] = True
                result.update(predicted_verdict=verdict, answer=value)
            results.append(result)
        oom = any("out of memory" in str(call.get("error", "")).lower() for call in call_report["batch_calls"])
        if oom:
            for item in items[offset + len(group) :]:
                results.append({
                    "fixture_id": item["fixture_id"],
                    "expected_verdict": item["expected_verdict_code"],
                    "status": "error",
                    "attempts": 0,
                    "predicted_verdict": None,
                    "error": "skipped_after_cuda_oom",
                })
            break
    total_seconds = time.perf_counter() - started
    metrics = {**score_judge_results(results), **_generation_totals(batch_reports)}
    metrics.update({
        "total_judge_seconds": total_seconds,
        "facts_per_second": len(items) / total_seconds if total_seconds > 0 else None,
        "batch_size_requested": batch_size,
    })
    return {
        "schema_version": "agent-pipeline-v2-judge-benchmark-result-v1",
        "batch_size": batch_size,
        "status": "error" if oom or metrics["parse_failure_count"] == len(items) else "partial" if metrics["parse_failure_count"] else "ok",
        "oom": oom,
        "items": results,
        "batch_reports": batch_reports,
        "metrics": metrics,
    }


def _number(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def aggregate_repeats(runs: list[dict]) -> dict:
    if not isinstance(runs, list) or not runs:
        raise ValueError("重复运行结果不能为空")
    batch_sizes = {run.get("batch_size") for run in runs}
    if len(batch_sizes) != 1:
        raise ValueError("只能聚合同一batch size")
    keys = set().union(*(run.get("metrics", {}) for run in runs))
    metrics = {}
    distributions = {}
    for key in sorted(keys):
        values = [run.get("metrics", {}).get(key) for run in runs]
        numeric = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        metrics[key] = statistics.median(numeric) if len(numeric) == len(values) and numeric else values[0] if all(value == values[0] for value in values) else None
        if numeric:
            distributions[key] = {"values": numeric, "min": min(numeric), "max": max(numeric), "median": statistics.median(numeric), "mean": statistics.mean(numeric), "stddev": statistics.pstdev(numeric) if len(numeric) > 1 else 0.0}
    statuses = [run.get("status") for run in runs]
    status = "error" if all(value == "error" for value in statuses) else "partial" if any(value != "ok" for value in statuses) else "ok"
    return {
        "schema_version": "agent-pipeline-v2-judge-benchmark-aggregate-v1",
        "batch_size": batch_sizes.pop(),
        "repeat_count": len(runs),
        "status": status,
        "oom": any(run.get("oom", False) for run in runs),
        "metrics": metrics,
        "distributions": distributions,
        "runs": runs,
    }


def benchmark_markdown(rows: list[dict]) -> str:
    lines = [
        "# Agent Pipeline v2 Judge Batching Benchmark",
        "",
        "| Batch | OOM | Peak allocated GiB | Peak reserved GiB | Total Judge s | Facts/s | Judge Accuracy |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda value: value["batch_size"]):
        metrics = row["metrics"]
        accuracy = metrics.get("judge_accuracy")
        lines.append(
            f"| {row['batch_size']} | {'yes' if row.get('oom') else 'no'} | {_number(metrics.get('peak_allocated_gib'))} | {_number(metrics.get('peak_reserved_gib'))} | {_number(metrics.get('total_judge_seconds'))} | {_number(metrics.get('facts_per_second'))} | {'N/A' if accuracy is None else f'{accuracy:.2%}'} |"
        )
    lines.extend(["", "Judge time includes tokenization, padding, generation, decoding, validation, and retries. Model loading and retrieval are excluded.", ""])
    return "\n".join(lines)
