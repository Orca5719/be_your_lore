"""Progressive, recall-aware metadata filtering before retrieval scoring."""

from __future__ import annotations

import re
import numpy as np

from retrieval import rank_chunks
from .lore_metadata import SCHEMA_VERSION, TIME_RE


DIMENSIONS = {
    "physical_rule",
    "character_knowledge",
    "character_relation",
    "identity",
    "space",
    "causality",
    "time",
    "world_rule",
}
DIMENSION_ALIASES = {
    "物理": "physical_rule",
    "生理": "physical_rule",
    "能力": "physical_rule",
    "认知": "character_knowledge",
    "知识": "character_knowledge",
    "关系": "character_relation",
    "身份": "identity",
    "空间": "space",
    "地点": "space",
    "因果": "causality",
    "时间": "time",
    "世界规则": "world_rule",
}


def _fact_text(fact: dict) -> str:
    values = [fact.get("normalized_fact"), fact.get("event"), fact.get("subject")]
    values.extend(fact.get("actors") or [])
    return " ".join(value for value in values if isinstance(value, str) and value.strip())


def extract_filter_hints(fact: dict, lore_metadata: dict) -> dict:
    if not isinstance(fact, dict):
        raise ValueError("fact必须为对象")
    vocabulary = lore_metadata.get("entity_vocabulary")
    if not isinstance(vocabulary, list):
        raise ValueError("lore metadata缺少entity_vocabulary")
    text = _fact_text(fact)
    explicit_subjects = []
    if isinstance(fact.get("subject"), str):
        explicit_subjects.append(fact["subject"])
    explicit_subjects.extend(value for value in (fact.get("actors") or []) if isinstance(value, str))
    subject_matches = {name for name in vocabulary if name in explicit_subjects}
    entities = sorted(subject_matches or {name for name in vocabulary if isinstance(name, str) and name and name in text})
    raw_dimension = fact.get("dimension") or fact.get("check_reason") or ""
    dimensions = {value for value in (fact.get("dimensions") or []) if value in DIMENSIONS}
    if raw_dimension in DIMENSIONS:
        dimensions.add(raw_dimension)
    if isinstance(raw_dimension, str):
        dimensions.update(value for key, value in DIMENSION_ALIASES.items() if key in raw_dimension)
        dimensions.update(value for value in DIMENSIONS if value in raw_dimension)
    times = sorted(set(TIME_RE.findall(text)))
    return {"entities": entities, "dimensions": sorted(dimensions), "times": times}


def _matches(row: dict, active: tuple[str, ...], hints: dict, mode: str = "all") -> bool:
    fields = {
        "entity": (set(row.get("entities") or []), set(hints["entities"])),
        "dimension": (set(row.get("dimension_tags") or []), set(hints["dimensions"])),
        "time": (set(row.get("time_markers") or []), set(hints["times"])),
    }
    matches = [bool(fields[name][0] & fields[name][1]) for name in active]
    return all(matches) if mode == "all" else any(matches)


def _levels(hints: dict) -> list[tuple[str, tuple[str, ...], str]]:
    available = tuple(name for name, field in (("entity", "entities"), ("dimension", "dimensions"), ("time", "times")) if hints[field])
    candidates = []
    if available:
        candidates.append(("+".join(available), available, "all"))
    without_time = tuple(name for name in ("entity", "dimension") if name in available)
    if without_time:
        candidates.append(("+".join(without_time), without_time, "all"))
    if len(without_time) == 2:
        candidates.append(("entity|dimension", without_time, "any"))
    elif len(without_time) == 1:
        candidates.append((without_time[0], without_time, "all"))
    elif "time" in available:
        candidates.append(("time", ("time",), "all"))
    candidates.append(("all", (), "all"))
    result = []
    seen = set()
    for label, active, mode in candidates:
        signature = (active, mode)
        if signature not in seen:
            seen.add(signature)
            result.append((label, active, mode))
    return result


def select_candidates(fact: dict, lore_metadata: dict, k: int = 5, min_candidates: int | None = None) -> dict:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k必须为正整数")
    if lore_metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("lore metadata schema_version无效")
    items = lore_metadata.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("lore metadata items不能为空")
    ids = [row.get("chunk_id") for row in items]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("lore metadata chunk id无效")
    if min_candidates is None:
        target = min(len(items), max(k * 4, 20))
    else:
        if isinstance(min_candidates, bool) or not isinstance(min_candidates, int) or min_candidates < k:
            raise ValueError("min_candidates必须是不小于k的正整数")
        target = min(len(items), min_candidates)
    hints = extract_filter_hints(fact, lore_metadata)
    trace = []
    chosen = None
    first_level = None
    for label, active, mode in _levels(hints):
        candidate_ids = ids[:] if not active else [row["chunk_id"] for row in items if _matches(row, active, hints, mode)]
        trace.append({"level": label, "candidate_count": len(candidate_ids)})
        if first_level is None:
            first_level = label
        if len(candidate_ids) >= target:
            chosen = (label, candidate_ids)
            break
    if chosen is None:  # all层必然满足target；此分支仅保护损坏输入。
        raise ValueError("metadata filter无法生成候选集")
    return {
        "schema_version": "agent-pipeline-v2.1-metadata-filter-v1",
        "hints": hints,
        "target_candidate_count": target,
        "selected_level": chosen[0],
        "candidate_ids": chosen[1],
        "candidate_count": len(chosen[1]),
        "fallback_used": chosen[0] != first_level,
        "trace": trace,
    }


def audit_oracle_fixture(fixture: dict, lore_metadata: dict, k: int = 5, min_candidates: int | None = None) -> dict:
    items = fixture.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Oracle fixture items不能为空")
    rows = []
    for item in items:
        selection = select_candidates(item["fact"], lore_metadata, k=k, min_candidates=min_candidates)
        expected = sorted(chunk["id"] for chunk in item["lore"])
        missing = sorted(set(expected) - set(selection["candidate_ids"]))
        rows.append({
            "fixture_id": item["fixture_id"],
            "expected_lore_ids": expected,
            "oracle_lore_retained": not missing,
            "missing_lore_ids": missing,
            "filter": selection,
        })
    hits = sum(row["oracle_lore_retained"] for row in rows)
    level_counts = {}
    for row in rows:
        level = row["filter"]["selected_level"]
        level_counts[level] = level_counts.get(level, 0) + 1
    sizes = [row["filter"]["candidate_count"] for row in rows]
    return {
        "schema_version": "agent-pipeline-v2.1-metadata-filter-audit-v1",
        "status": "ok",
        "k": k,
        "min_candidates": min_candidates,
        "facts_total": len(rows),
        "oracle_lore_pool_hits": hits,
        "oracle_lore_pool_recall": hits / len(rows),
        "selected_level_counts": dict(sorted(level_counts.items())),
        "candidate_count": {"min": min(sizes), "max": max(sizes), "mean": sum(sizes) / len(sizes)},
        "rows": rows,
    }


def rank_dense_with_metadata(
    vectors: np.ndarray,
    chunks: list[dict],
    query_vector: np.ndarray,
    fact: dict,
    lore_metadata: dict,
    k: int = 5,
    min_candidates: int | None = None,
) -> dict:
    chunk_ids = [chunk.get("id") for chunk in chunks]
    metadata_ids = [row.get("chunk_id") for row in lore_metadata.get("items", [])]
    if len(chunk_ids) != len(set(chunk_ids)) or set(chunk_ids) != set(metadata_ids):
        raise ValueError("lore metadata chunk集合与索引chunk不匹配，必须重建")
    selection = select_candidates(fact, lore_metadata, k=k, min_candidates=min_candidates)
    allowed = set(selection["candidate_ids"])
    indices = [index for index, chunk in enumerate(chunks) if chunk["id"] in allowed]
    subset_vectors = np.asarray(vectors)[indices]
    subset_chunks = [chunks[index] for index in indices]
    results = rank_chunks(subset_vectors, subset_chunks, query_vector, k=k)
    return {
        "schema_version": "agent-pipeline-v2.1-filtered-dense-result-v1",
        "method": "dense+metadata",
        "filter": selection,
        "scored_candidate_count": len(indices),
        "results": results,
    }


class FilteredDenseRetriever:
    """Adapter around the frozen Retriever; accepts a structured fact explicitly."""

    def __init__(self, retriever, lore_metadata: dict):
        if not hasattr(retriever, "vectors") or not hasattr(retriever, "metadata") or not hasattr(retriever, "encoder"):
            raise ValueError("retriever缺少vectors/metadata/encoder")
        self.retriever = retriever
        self.lore_metadata = lore_metadata

    def search_fact(self, fact: dict, query: str, k: int = 5, min_candidates: int | None = None) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("查询不能为空")
        vector = self.retriever.encoder.encode_queries([query])[0]
        return rank_dense_with_metadata(
            self.retriever.vectors,
            self.retriever.metadata["chunks"],
            vector,
            fact,
            self.lore_metadata,
            k=k,
            min_candidates=min_candidates,
        )
