"""Agent Pipeline v2 end-to-end orchestration."""

from __future__ import annotations

from .extractor import extract_events
from .judge import judge_events
from .report import build_report
from .retrieval import retrieve_events


def process_story(text: str, llm=None, retriever=None, top_k: int = 5, judge_batch_size: int = 1, device: str = "auto", progress=None) -> dict:
    extraction = extract_events(text, device=device, llm=llm, progress=progress)
    retrieval = retrieve_events(extraction, device=device, retriever=retriever, k=top_k, progress=progress)
    judge = judge_events(retrieval, batch_size=judge_batch_size, device=device, llm=llm, progress=progress)
    return build_report(judge)

