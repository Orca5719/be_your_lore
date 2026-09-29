"""Six-configuration retrieval benchmark for Agent Pipeline 2.1B."""

from __future__ import annotations

import time
import numpy as np
import statistics

from retrieval import rank_chunks

from .bm25 import BM25Retriever
from .contracts import RetrievalConfig
from .metadata_filter import select_candidates
from .rrf import fuse_rrf


CONFIGS = tuple(
    RetrievalConfig(method, enabled, 5)
    for method in ("dense", "bm25", "hybrid")
    for enabled in (False, True)
)


def aggregate_retrieval_runs(runs: list[dict], top_k: int) -> dict:
    if not isinstance(runs, list) or not runs:
        raise ValueError("retrieval重复结果不能为空")
    ids = [result["config_id"] for result in runs[0]["results"]]
    if any([result["config_id"] for result in run["results"]] != ids for run in runs):
        raise ValueError("retrieval重复结果配置不一致")
    aggregated = []
    for position, config_id in enumerate(ids):
        sources = [run["results"][position] for run in runs]
        keys = set().union(*(source["metrics"] for source in sources))
        metrics = {}
        distributions = {}
        for key in sorted(keys):
            values = [source["metrics"].get(key) for source in sources]
            numeric = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
            metrics[key] = statistics.median(numeric) if len(numeric) == len(values) and numeric else values[0] if all(value == values[0] for value in values) else None
            if numeric:
                distributions[key] = {"values": numeric, "median": statistics.median(numeric), "min": min(numeric), "max": max(numeric)}
        aggregated.append({
            "config_id": config_id,
            "method": sources[0]["method"],
            "metadata_filter": sources[0]["metadata_filter"],
            "status": "ok" if all(source.get("status") == "ok" for source in sources) else "partial",
            "metrics": metrics,
            "distributions": distributions,
            "rows": sources[0].get("rows", []),
        })
    encode_values = [run.get("query_encode_seconds") for run in runs]
    return {
        "schema_version": "agent-pipeline-v2.1-retrieval-benchmark-v1",
        "benchmark": "2.1B",
        "status": "ok" if all(result["status"] == "ok" for result in aggregated) else "partial",
        "top_k": top_k,
        "repeat_count": len(runs),
        "query_encode_seconds": statistics.median(encode_values),
        "query_encode_seconds_values": encode_values,
        "results": aggregated,
    }


def score_rows(rows: list[dict], k: int) -> dict:
    if not rows:
        raise ValueError("retrieval benchmark rows不能为空")
    complete = pool_complete = evidence_hits = evidence_total = 0
    reciprocal = []
    for row in rows:
        expected = set(row["expected_lore_ids"])
        retrieved = row["retrieved_ids"][:k]
        retrieved_set = set(retrieved)
        pool = set(row["candidate_ids"])
        complete += expected <= retrieved_set
        pool_complete += expected <= pool
        evidence_hits += len(expected & retrieved_set)
        evidence_total += len(expected)
        ranks = [index + 1 for index, chunk_id in enumerate(retrieved) if chunk_id in expected]
        reciprocal.append(1.0 / min(ranks) if ranks else 0.0)
    return {
        "facts_total": len(rows),
        "top_k": k,
        "complete_lore_hits": complete,
        "complete_lore_recall_at_k": complete / len(rows),
        "evidence_hits": evidence_hits,
        "evidence_total": evidence_total,
        "evidence_micro_recall_at_k": evidence_hits / evidence_total if evidence_total else None,
        "candidate_pool_complete_hits": pool_complete,
        "candidate_pool_complete_recall": pool_complete / len(rows),
        "mrr_at_k": sum(reciprocal) / len(reciprocal),
        "mean_candidate_count": sum(len(row["candidate_ids"]) for row in rows) / len(rows),
    }


def _dense_rank(vectors, chunks, query_vector, candidate_ids, depth):
    if candidate_ids is None:
        return rank_chunks(vectors, chunks, query_vector, k=depth)
    allowed = set(candidate_ids)
    indices = [index for index, chunk in enumerate(chunks) if chunk["id"] in allowed]
    return rank_chunks(np.asarray(vectors)[indices], [chunks[index] for index in indices], query_vector, k=depth)


def run_configuration(
    config: RetrievalConfig,
    fixture: dict,
    chunks: list[dict],
    vectors: np.ndarray,
    query_vectors: np.ndarray,
    bm25: BM25Retriever,
    lore_metadata: dict,
) -> dict:
    items = fixture.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("fixture items不能为空")
    if len(query_vectors) != len(items):
        raise ValueError("query vectors数量与fixture不匹配")
    started = time.perf_counter()
    rows = []
    for index, item in enumerate(items):
        fact = item["fact"]
        query = fact["normalized_fact"]
        selection = select_candidates(fact, lore_metadata, k=config.k) if config.metadata_filter else None
        candidate_ids = selection["candidate_ids"] if selection else [chunk["id"] for chunk in chunks]
        depth = min(len(candidate_ids), max(config.k * 4, 20))
        dense_rows = _dense_rank(vectors, chunks, query_vectors[index], candidate_ids if config.metadata_filter else None, depth) if config.method in {"dense", "hybrid"} else []
        lexical_rows = bm25.search(query, k=depth, candidate_ids=candidate_ids if config.metadata_filter else None) if config.method in {"bm25", "hybrid"} else []
        if config.method == "dense":
            results = dense_rows[: config.k]
        elif config.method == "bm25":
            results = lexical_rows[: config.k]
        else:
            results = fuse_rrf(dense_rows, lexical_rows, k=config.k)["results"]
        rows.append({
            "fixture_id": item["fixture_id"],
            "fact": fact,
            "query": query,
            "expected_lore_ids": sorted(chunk["id"] for chunk in item["lore"]),
            "candidate_ids": candidate_ids,
            "filter": selection,
            "retrieved_ids": [row["id"] for row in results],
            "results": results,
        })
    elapsed = time.perf_counter() - started
    metrics = score_rows(rows, config.k)
    metrics.update({"ranking_seconds": elapsed, "facts_per_second": len(rows) / elapsed if elapsed else None})
    return {
        "schema_version": "agent-pipeline-v2.1-retrieval-config-result-v1",
        "config_id": f"{config.method}__metadata_{'on' if config.metadata_filter else 'off'}",
        "method": config.method,
        "metadata_filter": config.metadata_filter,
        "top_k": config.k,
        "status": "ok",
        "metrics": metrics,
        "rows": rows,
    }


def _percent(value):
    return "N/A" if value is None else f"{value:.2%}"


def render_retrieval_markdown(report: dict) -> str:
    k = report["top_k"]
    lines = [
        "# Benchmark 2.1B — Retrieval A/B",
        "",
        f"| Method | Metadata | Complete Recall@{k} | Evidence Recall@{k} | MRR@{k} | Candidate Pool Recall | Mean Candidates | Ranking s |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["results"]:
        metrics = result["metrics"]
        lines.append(
            f"| {result['method']} | {'on' if result['metadata_filter'] else 'off'} | {_percent(metrics['complete_lore_recall_at_k'])} | {_percent(metrics['evidence_micro_recall_at_k'])} | {metrics['mrr_at_k']:.4f} | {_percent(metrics['candidate_pool_complete_recall'])} | {metrics['mean_candidate_count']:.2f} | {metrics['ranking_seconds']:.4f} |"
        )
    lines.extend(["", "Complete Recall requires every Oracle lore item for a fact to appear within Top-K.", ""])
    return "\n".join(lines)
