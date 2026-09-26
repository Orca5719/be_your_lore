from __future__ import annotations

from collections import Counter
from typing import Callable

from .schema import stable_digest


CELLS = {
    "dense_v1": ("dense", "v1"),
    "hybrid_v1": ("hybrid", "v1"),
    "dense_v21": ("dense", "v2.1"),
    "hybrid_v21": ("hybrid", "v2.1"),
}


def _call_retriever(fn: Callable, extraction: dict, top_k: int):
    return fn(extraction, top_k=top_k)


def _call_judge(fn: Callable, retrieval: dict, batch_size: int):
    return fn(retrieval, batch_size=batch_size)


def _cell_summary(cell: dict) -> dict:
    if cell["status"] != "ok":
        return {"status": cell["status"], "events": 0, "contradictions": 0}
    result = cell["result"]
    nested = result.get("judge", {})
    items = result.get("items")
    if not isinstance(items, list):
        items = nested.get("items", []) if isinstance(nested, dict) else []
    return {
        "status": "ok",
        "events": len(items),
        "contradictions": sum(item.get("verdict") == "contradiction" for item in items),
    }


def run_paired_matrix(
    extraction_rows: list[dict],
    dense: Callable,
    hybrid: Callable,
    judge_v1: Callable,
    judge_v21: Callable,
    *,
    top_k: int = 5,
    batch_size: int = 8,
) -> dict:
    if top_k != 5 or batch_size != 8:
        raise ValueError("Benchmark 3 requires Top-K=5 and Judge batch=8")
    event_digest = stable_digest(extraction_rows)
    extraction = {"events": extraction_rows}
    retrievals = {}
    retrieval_errors = {}
    for name, fn in (("dense", dense), ("hybrid", hybrid)):
        try:
            retrievals[name] = _call_retriever(fn, extraction, top_k)
        except Exception as exc:
            retrieval_errors[name] = f"{type(exc).__name__}: {exc}"
    cells = {}
    judges = {"v1": judge_v1, "v2.1": judge_v21}
    for cell_name, (retrieval_name, judge_name) in CELLS.items():
        base = {
            "retrieval": retrieval_name,
            "judge": judge_name,
            "event_digest": event_digest,
        }
        if retrieval_name in retrieval_errors:
            cells[cell_name] = dict(base, status="error", error=retrieval_errors[retrieval_name], result={})
            continue
        try:
            result = _call_judge(judges[judge_name], retrievals[retrieval_name], batch_size)
            cells[cell_name] = dict(base, status="ok", result=result)
        except Exception as exc:
            cells[cell_name] = dict(base, status="error", error=f"{type(exc).__name__}: {exc}", result={})
    status_counts = Counter(cell["status"] for cell in cells.values())
    status = "ok" if status_counts["ok"] == 4 else "error" if status_counts["error"] == 4 else "partial"
    summaries = {name: _cell_summary(cell) for name, cell in cells.items()}
    return {
        "schema_version": "agent-pipeline-v3-paired-v1",
        "status": status,
        "event_digest": event_digest,
        "cells": cells,
        "summaries": summaries,
        "deltas": {
            "retrieval_only_v1": ["dense_v1", "hybrid_v1"],
            "retrieval_only_v21": ["dense_v21", "hybrid_v21"],
            "judge_only_dense": ["dense_v1", "dense_v21"],
            "judge_only_hybrid": ["hybrid_v1", "hybrid_v21"],
            "combined": ["dense_v1", "hybrid_v21"],
        },
    }
