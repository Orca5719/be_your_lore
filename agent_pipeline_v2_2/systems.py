from __future__ import annotations

import time

from .schema import SystemConfig, validate_system_config
from .story_pipeline import PipelineSystem, StageRun


def _measure(call):
    started = time.perf_counter()
    value = call()
    return value, time.perf_counter() - started


def _baseline_dependencies():
    from agent_pipeline_v2.extractor import extract_events
    from agent_pipeline_v2.retrieval import retrieve_events
    from agent_pipeline_v2.judge import judge_events
    from agent_pipeline_v2.report import build_report
    return {"extract": extract_events, "retrieve": retrieve_events, "judge": judge_events, "report": build_report}


def _candidate_dependencies():
    from agent_pipeline_v2.extractor import extract_events
    from agent_pipeline_v2_1.extractor import ExtractionRepairLLM
    from agent_pipeline_v2_1.story_pipeline import retrieve_story_events, judge_story_events, build_story_report
    return {"repair": ExtractionRepairLLM, "extract": extract_events, "retrieve": retrieve_story_events, "judge": judge_story_events, "report": build_story_report}


def build_baseline_system(llm, retriever, device: str = "auto", dependencies: dict | None = None) -> PipelineSystem:
    config = validate_system_config(SystemConfig.baseline())
    deps = _baseline_dependencies() if dependencies is None else dependencies

    def run(text: str, progress=None):
        extraction, t1 = _measure(lambda: deps["extract"](text, device=device, llm=llm, progress=progress))
        retrieval, t2 = _measure(lambda: deps["retrieve"](extraction, device=device, retriever=retriever, k=config.top_k, progress=progress))
        judge, t3 = _measure(lambda: deps["judge"](retrieval, batch_size=config.judge_batch_size, device=device, llm=llm, progress=progress))
        report, t4 = _measure(lambda: deps["report"](judge))
        return StageRun(extraction, retrieval, judge, report, {"extraction": t1, "retrieval": t2, "judge": t3, "report": t4})

    return PipelineSystem(config, run, device)


def build_candidate_system(llm, retriever, device: str = "auto", dependencies: dict | None = None) -> PipelineSystem:
    config = validate_system_config(SystemConfig.candidate())
    deps = _candidate_dependencies() if dependencies is None else dependencies

    def run(text: str, progress=None):
        repaired = deps["repair"](llm)
        extraction, t1 = _measure(lambda: deps["extract"](text, device=device, llm=repaired, progress=progress))
        retrieval, t2 = _measure(lambda: deps["retrieve"](extraction, retriever, config.retrieval, config.metadata_filter, config.top_k, progress))
        judge, t3 = _measure(lambda: deps["judge"](retrieval, llm, config.judge_batch_size, progress))
        report, t4 = _measure(lambda: deps["report"](judge))
        return StageRun(extraction, retrieval, judge, report, {"extraction": t1, "retrieval": t2, "judge": t3, "report": t4})

    return PipelineSystem(config, run, device)
