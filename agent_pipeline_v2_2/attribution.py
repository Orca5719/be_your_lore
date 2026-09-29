from __future__ import annotations

from collections import Counter

from .scoring import _case_state


def attribute_first_failures(cases: list[dict], run_rows: list[dict], review: dict, k: int = 5) -> dict:
    by_case = {row["case_id"]: row for row in run_rows}
    details = []
    for case in cases:
        case_id = case["id"]
        row = by_case.get(case_id)
        conflict_facts = [x for x in case.get("gold_facts", []) if x.get("expected_verdict") == "矛盾"]
        if row is None or row.get("status") == "error":
            for gold in conflict_facts:
                details.append({"case_id": case_id, "gold_fact_id": gold["id"], "primary_cause": "execution_failure", "event_ids": [], "evidence_ids": []})
            continue
        states, _ = _case_state(case, row, review, k)
        reportable = {
            gold_id: finding["id"]
            for finding in case.get("gold_findings", [])
            for gold_id in finding.get("gold_fact_ids", [])
        }
        finding_mapping = review.get("finding_mapping", {}).get(case_id, {})
        for state in states:
            gold = state["gold"]
            if gold.get("expected_verdict") != "矛盾":
                continue
            if not state["mapped"]:
                cause = "extraction_miss"
            elif state["alternatives"] and not state["retrieval_ok"]:
                cause = "retrieval_miss"
            elif "contradiction" not in state["predictions"]:
                cause = "judge_fn"
            elif gold["id"] in reportable and not finding_mapping.get(reportable[gold["id"]], []):
                cause = "report_miss"
            else:
                continue
            details.append({
                "case_id": case_id,
                "gold_fact_id": gold["id"],
                "primary_cause": cause,
                "event_ids": list(state["mapped"]),
                "evidence_ids": [item for group in state["alternatives"] for item in group],
            })
    counts = Counter(item["primary_cause"] for item in details)
    return {"items": details, "counts": dict(sorted(counts.items()))}
