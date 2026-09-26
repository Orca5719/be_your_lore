"""Rank-only fusion for dense and BM25 retrieval results."""

from __future__ import annotations

import copy


def _validate_rows(rows: list[dict], name: str) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"{name}结果必须是数组")
    ids = [row.get("id") for row in rows if isinstance(row, dict)]
    if len(ids) != len(rows) or any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{name}结果ID无效或重复")


def fuse_rrf(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 5,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> dict:
    _validate_rows(dense_results, "dense")
    _validate_rows(bm25_results, "BM25")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k必须为正整数")
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 1:
        raise ValueError("rrf_k必须为正整数")
    for label, value in (("dense_weight", dense_weight), ("bm25_weight", bm25_weight)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(label + "必须为正数")

    entries = {}
    for source, rows, weight in (("dense", dense_results, float(dense_weight)), ("bm25", bm25_results, float(bm25_weight))):
        for rank, row in enumerate(rows, 1):
            chunk_id = row["id"]
            if chunk_id not in entries:
                entries[chunk_id] = {
                    "base": copy.deepcopy(row),
                    "dense_rank": None,
                    "bm25_rank": None,
                    "dense_score": None,
                    "bm25_score": None,
                    "dense_contribution": 0.0,
                    "bm25_contribution": 0.0,
                }
            elif row.get("text") != entries[chunk_id]["base"].get("text"):
                raise ValueError("同一chunk ID在两路结果中的原文不一致")
            entry = entries[chunk_id]
            entry[source + "_rank"] = rank
            raw_score = row.get("lexical_score", row.get("score")) if source == "bm25" else row.get("score")
            entry[source + "_score"] = raw_score
            entry[source + "_contribution"] = weight / (rrf_k + rank)

    rows = []
    for chunk_id, entry in entries.items():
        base = entry.pop("base")
        base.pop("rank", None)
        base.pop("matched_terms", None)
        base.pop("lexical_score", None)
        base.pop("score", None)
        rrf_score = entry["dense_contribution"] + entry["bm25_contribution"]
        sources = [name for name in ("dense", "bm25") if entry[name + "_rank"] is not None]
        rows.append({
            **base,
            "score": rrf_score,
            "rrf_score": rrf_score,
            **entry,
            "sources": sources,
        })
    rows.sort(key=lambda row: (-row["rrf_score"], min(value for value in (row["dense_rank"], row["bm25_rank"]) if value is not None), row["id"]))
    selected = rows[: min(k, len(rows))]
    for rank, row in enumerate(selected, 1):
        row["rank"] = rank
    return {
        "schema_version": "agent-pipeline-v2.1-rrf-result-v1",
        "method": "dense+bm25-rrf",
        "config": {"rrf_k": rrf_k, "dense_weight": float(dense_weight), "bm25_weight": float(bm25_weight)},
        "dense_input_count": len(dense_results),
        "bm25_input_count": len(bm25_results),
        "union_count": len(rows),
        "results": selected,
    }

