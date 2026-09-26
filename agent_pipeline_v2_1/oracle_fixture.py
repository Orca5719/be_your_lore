"""Build the 72-item Oracle Retrieval fixture from reviewed story gold facts."""

from __future__ import annotations

from collections import Counter
import copy


VERDICT_CODES = {"一致": "consistent", "矛盾": "contradiction", "不确定": "uncertain"}
SCHEMA_VERSION = "agent-pipeline-v2.1-oracle-fixture-v1"


def _gold_lore_ids(fact: dict) -> list[str]:
    groups = fact.get("minimum_evidence_sets", [])
    if not groups or not groups[0]:
        raise ValueError(f"{fact.get('id')}缺少minimum_evidence_sets")
    return list(dict.fromkeys(groups[0]))


def build_oracle_fixture(dataset: dict, index_metadata: dict, curation: dict) -> dict:
    chunks = {chunk["id"]: chunk for chunk in index_metadata.get("chunks", [])}
    curated = curation.get("facts", {})
    overrides = curation.get("verdict_overrides", {})
    items = []
    flags = []
    for case in dataset.get("cases", []):
        for fact in case.get("gold_facts", []):
            normalized_fact = fact.get("normalized_fact")
            original_expected = VERDICT_CODES.get(fact.get("expected_verdict"))
            override = overrides.get(normalized_fact)
            expected = override.get("verdict") if isinstance(override, dict) else original_expected
            if expected is None:
                raise ValueError(f"{case['id']}/{fact.get('id')} verdict无效")
            fixture_id = f"OR-{case['id']}-{fact['id']}"
            if original_expected == "uncertain":
                selection = curated.get(normalized_fact)
                if not isinstance(selection, dict) or not selection.get("lore_ids"):
                    raise ValueError(f"uncertain fact {fixture_id} 必须配置非空lore")
                lore_ids = list(dict.fromkeys(selection["lore_ids"]))
                oracle_selection = {"source": "manual_label_override" if override else "manual_curation", "rationale": override.get("rationale", "") if override else selection.get("rationale", "")}
                if not oracle_selection["rationale"].strip():
                    raise ValueError(f"uncertain fact {fixture_id} 缺少rationale")
                if selection.get("review_flag"):
                    flags.append({"fixture_id": fixture_id, "reason": selection["review_flag"]})
            else:
                lore_ids = _gold_lore_ids(fact)
                oracle_selection = {
                    "source": "minimum_evidence_set",
                    "rationale": "使用gold fact标注的第一组充分证据。",
                }
            missing = [chunk_id for chunk_id in lore_ids if chunk_id not in chunks]
            if missing:
                raise ValueError(f"{fixture_id}引用不存在的lore：{missing}")
            items.append({
                "fixture_id": fixture_id,
                "case_id": case["id"],
                "gold_fact_id": fact["id"],
                "expected_verdict": expected,
                "fact": {key: copy.deepcopy(fact.get(key)) for key in ("subject", "predicate", "object", "normalized_fact", "type", "dimension", "source_anchors", "context_anchors")},
                "story_context": case["story"],
                "lore": [copy.deepcopy(chunks[chunk_id]) for chunk_id in lore_ids],
                "oracle_selection": oracle_selection,
            })
    counts = Counter(item["expected_verdict"] for item in items)
    expected_counts = Counter(curation.get("expected_counts", {}))
    if len(items) != 72 or counts != expected_counts:
        raise ValueError(f"Oracle fixture数量或类别分布无效，预期{dict(expected_counts)}，实际{dict(counts)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "review_required" if flags else "reviewed",
        "source_dataset": {"name": dataset.get("name"), "version": dataset.get("version")},
        "index_version": dataset.get("corpus_snapshot", {}).get("index_version"),
        "counts": dict(counts),
        "review_summary": {"flagged_count": len(flags), "flags": flags},
        "items": items,
    }
