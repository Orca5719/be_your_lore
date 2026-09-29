from __future__ import annotations

from collections import Counter, defaultdict


LABELS = {"一致": "consistent", "矛盾": "contradiction", "不确定": "uncertain"}


def ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def prf(tp: int, fp: int, fn: int) -> dict:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def _case_state(case: dict, row: dict, review: dict, k: int):
    case_id = case["id"]
    result = row.get("result", {})
    judge = result.get("judge", {})
    events = {x["id"]: x for x in judge.get("events", []) if isinstance(x, dict) and isinstance(x.get("id"), str)}
    items = {x["event_id"]: x for x in judge.get("items", []) if isinstance(x, dict) and isinstance(x.get("event_id"), str)}
    retrieved = {x["event_id"]: x for x in judge.get("retrieval", {}).get("items", []) if isinstance(x, dict) and isinstance(x.get("event_id"), str)}
    mapping = review.get("event_mapping", {}).get(case_id, {})
    labels = review.get("system_event_labels", {}).get(case_id, {})
    valid = {eid for eid, value in labels.items() if value.get("label") == "valid_checkable"}
    facts = []
    for gold in case.get("gold_facts", []):
        mapped = [eid for eid in mapping.get(gold["id"], []) if eid in events and (not labels or eid in valid)]
        alternatives = gold.get("acceptable_evidence_sets", []) or gold.get("minimum_evidence_sets", [])
        retrieval_ok = not alternatives
        retrieval_ready = False
        for eid in mapped:
            retrieval_ready |= retrieved.get(eid, {}).get("status") == "ok"
            found = {x.get("id") for x in retrieved.get(eid, {}).get("evidence", [])[:k] if isinstance(x, dict)}
            retrieval_ok |= any(set(group) <= found for group in alternatives)
        predictions = {items[eid].get("verdict") for eid in mapped if items.get(eid, {}).get("status") == "ok"}
        facts.append({
            "gold": gold,
            "mapped": mapped,
            "alternatives": alternatives,
            "retrieval_ready": retrieval_ready,
            "retrieval_ok": retrieval_ok,
            "predictions": predictions,
        })
    return facts, len(valid or events)


def score_system(cases: list[dict], run_rows: list[dict], review: dict, k: int = 5) -> dict:
    by_case = {row["case_id"]: row for row in run_rows}
    extraction_hits = extraction_total = 0
    retrieval_hits = retrieval_eligible = 0
    judge_correct = judge_eligible = judge_fp = judge_fn_evidence = 0
    conflict_total = conflict_tp = predicted_fp = 0
    confusion = defaultdict(Counter)
    unsupported = emitted = 0
    for case in cases:
        row = by_case.get(case["id"], {})
        facts, event_count = _case_state(case, row, review, k)
        emitted += event_count
        unsupported_labels = review.get("reasoning_support_labels", {}).get(case["id"], {})
        unsupported += sum(value.get("label") == "unsupported" for value in unsupported_labels.values())
        for state in facts:
            gold = state["gold"]
            extracted = bool(state["mapped"])
            extraction_total += 1
            extraction_hits += extracted
            if extracted and state["alternatives"]:
                retrieval_eligible += 1
                retrieval_hits += state["retrieval_ok"]
            expected = LABELS[gold["expected_verdict"]]
            eligible = extracted and state["retrieval_ready"] and state["retrieval_ok"]
            if eligible:
                judge_eligible += 1
                predicted = next(iter(state["predictions"])) if len(state["predictions"]) == 1 else "mixed_or_missing"
                judge_correct += expected in state["predictions"]
                confusion[expected][predicted] += 1
            predicted_conflict = "contradiction" in state["predictions"]
            if expected == "contradiction":
                conflict_total += 1
                conflict_tp += predicted_conflict
                judge_fn_evidence += eligible and not predicted_conflict
            elif predicted_conflict:
                predicted_fp += 1
                judge_fp += 1
    conflict_fn = conflict_total - conflict_tp
    extraction_misses = extraction_total - extraction_hits
    retrieval_misses = retrieval_eligible - retrieval_hits
    return {
        "extraction": {"hits": extraction_hits, "misses": extraction_misses, "total": extraction_total, "recall": ratio(extraction_hits, extraction_total)},
        "retrieval": {"hits": retrieval_hits, "misses": retrieval_misses, "eligible": retrieval_eligible, "k": k, "recall_at_5": ratio(retrieval_hits, retrieval_eligible)},
        "judge": {"correct": judge_correct, "eligible": judge_eligible, "accuracy": ratio(judge_correct, judge_eligible), "fp": judge_fp, "fn_total": conflict_fn, "fn_with_complete_evidence": judge_fn_evidence, "confusion": {key: dict(value) for key, value in confusion.items()}},
        "unsupported_reasoning": {"count": unsupported, "eligible": emitted, "rate": ratio(unsupported, emitted)},
        "end_to_end_conflict": prf(conflict_tp, predicted_fp, conflict_fn),
    }
