"""Stable public configuration contracts for the 2.1 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RetrievalMethod = Literal["dense", "bm25", "hybrid"]


@dataclass(frozen=True)
class BenchmarkConfig:
    judge_batch_size: int = 8
    repeats: int = 3
    report_namespace: str = "agent_pipeline_v2_1"


@dataclass(frozen=True)
class RetrievalConfig:
    method: RetrievalMethod
    metadata_filter: bool
    k: int

    def __post_init__(self) -> None:
        if self.method not in {"dense", "bm25", "hybrid"}:
            raise ValueError("method必须为dense/bm25/hybrid")
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValueError("k必须为正整数")
        if type(self.metadata_filter) is not bool:
            raise ValueError("metadata_filter必须为布尔值")
