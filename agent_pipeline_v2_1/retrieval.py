"""Unified v2.1 retrieval surface for dense, BM25, and RRF hybrid."""

from __future__ import annotations

import numpy as np

from retrieval import rank_chunks

from .bm25 import BM25Retriever
from .metadata_filter import select_candidates
from .rrf import fuse_rrf


METHODS = {"dense", "bm25", "hybrid"}


class V21Retriever:
    def __init__(self, dense_retriever, lore_metadata: dict):
        if not hasattr(dense_retriever, "vectors") or not hasattr(dense_retriever, "metadata") or not hasattr(dense_retriever, "encoder"):
            raise ValueError("dense_retriever缺少vectors/metadata/encoder")
        chunks = dense_retriever.metadata.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("dense索引缺少chunks")
        metadata_ids = {row.get("chunk_id") for row in lore_metadata.get("items", [])}
        chunk_ids = {row.get("id") for row in chunks}
        if metadata_ids != chunk_ids:
            raise ValueError("lore metadata与dense索引chunk不匹配")
        self.dense = dense_retriever
        self.chunks = chunks
        self.lore_metadata = lore_metadata
        self.bm25 = BM25Retriever(chunks)

    def _dense_rank(self, query: str, candidate_ids: list[str] | None, depth: int) -> list[dict]:
        vector = self.dense.encoder.encode_queries([query])[0]
        if candidate_ids is None:
            return rank_chunks(self.dense.vectors, self.chunks, vector, k=depth)
        allowed = set(candidate_ids)
        indices = [index for index, chunk in enumerate(self.chunks) if chunk["id"] in allowed]
        vectors = np.asarray(self.dense.vectors)[indices]
        chunks = [self.chunks[index] for index in indices]
        return rank_chunks(vectors, chunks, vector, k=depth)

    def search_fact(
        self,
        fact: dict,
        query: str,
        method: str = "dense",
        metadata_filter: bool = False,
        k: int = 5,
        rrf_depth: int | None = None,
    ) -> dict:
        if method not in METHODS:
            raise ValueError("method必须为dense/bm25/hybrid")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("查询不能为空")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k必须为正整数")
        selection = select_candidates(fact, self.lore_metadata, k=k) if metadata_filter else None
        candidate_ids = selection["candidate_ids"] if selection else None
        candidate_count = len(candidate_ids) if candidate_ids is not None else len(self.chunks)
        if rrf_depth is None:
            depth = min(candidate_count, max(k * 4, 20))
        else:
            if isinstance(rrf_depth, bool) or not isinstance(rrf_depth, int) or rrf_depth < k:
                raise ValueError("rrf_depth必须是不小于k的正整数")
            depth = min(candidate_count, rrf_depth)
        dense_rows = self._dense_rank(query, candidate_ids, depth) if method in {"dense", "hybrid"} else []
        bm25_report = self.bm25.search_with_report(query, k=depth, candidate_ids=candidate_ids) if method in {"bm25", "hybrid"} else None
        bm25_rows = bm25_report["results"] if bm25_report else []
        fusion = None
        if method == "dense":
            results = dense_rows[:k]
        elif method == "bm25":
            results = bm25_rows[:k]
        else:
            fusion = fuse_rrf(dense_rows, bm25_rows, k=k)
            results = fusion["results"]
        return {
            "schema_version": "agent-pipeline-v2.1-retrieval-result-v1",
            "method": method,
            "metadata_filter_enabled": metadata_filter,
            "filter": selection,
            "query": query,
            "top_k": k,
            "ranker_depth": depth,
            "candidate_count": candidate_count,
            "dense_input_count": len(dense_rows),
            "bm25_input_count": len(bm25_rows),
            "bm25": bm25_report,
            "fusion": fusion,
            "results": results,
        }

