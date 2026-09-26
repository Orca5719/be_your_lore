"""Deterministic Judge-only fixture construction from the Fact-Level corpus."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json


VERDICTS = ("一致", "矛盾", "不确定")
VERDICT_CODES = {"一致": "consistent", "矛盾": "contradiction", "不确定": "uncertain"}


def _digest(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_claims(dataset: dict, per_label: int = 16) -> list[dict]:
    if isinstance(per_label, bool) or not isinstance(per_label, int) or per_label < 1:
        raise ValueError("per_label必须为正整数")
    candidates = defaultdict(list)
    for case in dataset.get("cases", []):
        if case.get("prior_development_semantic_overlap"):
            continue
        for claim_index, claim in enumerate(case.get("claims", []), 1):
            verdict = claim.get("expected_verdict")
            if verdict not in VERDICTS:
                continue
            candidates[verdict].append({
                "fixture_id": f"{case['id']}-C{claim_index}",
                "source_case_id": case["id"],
                "claim_index": claim_index,
                "group": case.get("group"),
                "primary_dimension": case.get("primary_dimension", "未分类"),
                "family_id": case.get("family_id", case["id"]),
                "story_context": case["text"],
                "fact": claim["input_quote"],
                "actors": list(claim.get("subject_any", [])),
                "expected_verdict": verdict,
                "annotated_evidence": list(claim.get("evidence", [])),
                "required_context": claim.get("required_context"),
            })
    selected = []
    for verdict in VERDICTS:
        by_dimension = defaultdict(list)
        for row in candidates[verdict]:
            by_dimension[row["primary_dimension"]].append(row)
        queues = {
            dimension: deque(sorted(rows, key=lambda row: (row["family_id"], row["source_case_id"], row["claim_index"])))
            for dimension, rows in by_dimension.items()
        }
        dimensions = sorted(queues)
        chosen = []
        while len(chosen) < per_label and any(queues[dimension] for dimension in dimensions):
            for dimension in dimensions:
                if queues[dimension] and len(chosen) < per_label:
                    chosen.append(queues[dimension].popleft())
        if len(chosen) != per_label:
            raise ValueError(f"{verdict}候选不足{per_label}条")
        selected.extend(chosen)
    return selected


def _canonical_chunks(claim: dict, chunks: list[dict]) -> list[dict]:
    found = []
    for annotation in claim["annotated_evidence"]:
        matches = []
        for chunk in chunks:
            same_file = chunk.get("file") == annotation.get("file")
            overlaps = chunk.get("start_line", 0) <= annotation.get("end_line", 0) and chunk.get("end_line", 0) >= annotation.get("start_line", 0)
            quote = annotation.get("quote", "")
            text = chunk.get("text", "")
            if same_file and overlaps and quote and (quote in text or text in quote):
                matches.append(chunk)
        if not matches:
            raise ValueError(f"标注证据无法映射到索引：{claim['fixture_id']}")
        found.append(matches[0])
    result = []
    seen = set()
    for chunk in found:
        if chunk["id"] not in seen:
            result.append(chunk)
            seen.add(chunk["id"])
    return result


def build_fixture(dataset: dict, index_metadata: dict, retriever, per_label: int = 16, k: int = 5) -> dict:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k必须为正整数")
    chunks = index_metadata.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("索引metadata缺少chunks")
    selected = select_claims(dataset, per_label=per_label)
    items = []
    for claim in selected:
        canonical = _canonical_chunks(claim, chunks)
        query = claim["fact"] + (("\n条件：" + claim["required_context"]) if claim.get("required_context") else "")
        retrieved = retriever.search(query, k=k)
        lore = []
        seen = set()
        for chunk in canonical + list(retrieved):
            if chunk["id"] in seen:
                continue
            lore.append({key: chunk[key] for key in ("id", "text", "file", "start_line", "end_line", "heading_path") if key in chunk} | ({"score": chunk["score"]} if "score" in chunk else {}))
            seen.add(chunk["id"])
            if len(lore) == k:
                break
        if not lore:
            raise ValueError(f"Judge fixture没有候选lore：{claim['fixture_id']}")
        item = {key: value for key, value in claim.items() if key != "annotated_evidence"}
        item.update({
            "expected_verdict_code": VERDICT_CODES[claim["expected_verdict"]],
            "canonical_evidence_ids": [chunk["id"] for chunk in canonical],
            "lore": lore,
        })
        items.append(item)
    fixture = {
        "schema_version": "agent-pipeline-v2-judge-fixture-v1",
        "name": "Agent Pipeline v2 Judge Batching Fixture",
        "status": "annotation_draft_not_final_benchmark",
        "source_dataset_name": dataset.get("name"),
        "source_dataset_digest": _digest(dataset),
        "index_version": index_metadata.get("version") or index_metadata.get("index_version"),
        "k": k,
        "per_label": per_label,
        "selection_policy": "exclude prior development semantic overlap; stable round-robin by verdict and primary dimension",
        "items": items,
    }
    fixture["items_digest"] = _digest(items)
    validate_fixture(fixture, expected_per_label=per_label, expected_k=k)
    return fixture


def validate_fixture(fixture: dict, expected_per_label: int = 16, expected_k: int = 5) -> None:
    if not isinstance(fixture, dict) or fixture.get("schema_version") != "agent-pipeline-v2-judge-fixture-v1":
        raise ValueError("Judge fixture schema无效")
    items = fixture.get("items")
    if not isinstance(items, list) or len(items) != expected_per_label * len(VERDICTS):
        raise ValueError("Judge fixture条目数量无效")
    ids = [item.get("fixture_id") for item in items]
    if any(not isinstance(item_id, str) or not item_id for item_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Judge fixture ID无效或重复")
    counts = Counter(item.get("expected_verdict") for item in items)
    if counts != Counter({verdict: expected_per_label for verdict in VERDICTS}):
        raise ValueError("Judge fixture标签不平衡")
    for item in items:
        lore = item.get("lore")
        if not isinstance(lore, list) or not lore or len(lore) > expected_k:
            raise ValueError("Judge fixture lore数量无效")
        lore_ids = [chunk.get("id") for chunk in lore]
        if any(not isinstance(chunk_id, str) for chunk_id in lore_ids) or len(lore_ids) != len(set(lore_ids)):
            raise ValueError("Judge fixture lore ID无效或重复")
        canonical = item.get("canonical_evidence_ids")
        if not isinstance(canonical, list) or not set(canonical) <= set(lore_ids):
            raise ValueError("标注证据未包含在Judge lore中")
        if item["expected_verdict"] != "不确定" and not canonical:
            raise ValueError("一致/矛盾fixture必须包含标注证据")
        if item.get("expected_verdict_code") != VERDICT_CODES[item["expected_verdict"]]:
            raise ValueError("中英文Judge标签不匹配")
    if fixture.get("items_digest") != _digest(items):
        raise ValueError("Judge fixture items_digest不匹配")
