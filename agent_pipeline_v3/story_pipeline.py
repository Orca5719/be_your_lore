from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from .schema import STAGES, SystemConfig, validate_run_row


@dataclass
class StageRun:
    extraction: dict
    retrieval: dict
    judge: dict
    report: dict
    seconds: dict[str, float]


@dataclass
class PipelineSystem:
    config: SystemConfig
    runner: Callable
    device: str = "auto"

    def run_stages(self, text: str, progress=None):
        return self.runner(text, progress=progress)


def _empty_metrics(total: float) -> dict:
    return {name: {"seconds": total if name == "total" else 0.0} for name in STAGES}


def process_story(system: PipelineSystem, case_id: str, text: str, progress=None) -> dict:
    started = time.perf_counter()
    cuda = None
    try:
        if system.device == "cuda":
            import torch
            if torch.cuda.is_available():
                cuda = torch
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
        raw = system.run_stages(text, progress=progress)
        total = time.perf_counter() - started
        if isinstance(raw, StageRun):
            metrics = {name: {"seconds": float(raw.seconds[name])} for name in STAGES[:-1]}
            metrics["total"] = {"seconds": total}
            extraction, retrieval, judge, report = raw.extraction, raw.retrieval, raw.judge, raw.report
        else:
            extraction, retrieval, judge, report = raw
            metrics = _empty_metrics(total)
        result = dict(report)
        result.setdefault("judge", judge)
        result["retrieval_provenance"] = {
            "method": system.config.retrieval,
            "metadata_filter": system.config.metadata_filter,
            "top_k": system.config.top_k,
        }
        result["benchmark_stages"] = {"extraction": extraction, "retrieval": retrieval, "judge": judge}
        status = report.get("status", "error")
        status = status if status in {"ok", "partial", "error", "pending_review"} else "error"
        row = {"case_id": case_id, "system": system.config.name, "status": status, "result": result, "stage_metrics": metrics}
    except Exception as exc:
        total = time.perf_counter() - started
        row = {"case_id": case_id, "system": system.config.name, "status": "error", "result": {}, "stage_metrics": _empty_metrics(total), "error": f"{type(exc).__name__}: {exc}"}
    if cuda is not None:
        cuda.cuda.synchronize()
        row["cuda"] = {
            "peak_allocated_bytes": cuda.cuda.max_memory_allocated(),
            "peak_reserved_bytes": cuda.cuda.max_memory_reserved(),
        }
    validate_run_row(row)
    return row
