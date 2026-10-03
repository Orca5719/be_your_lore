"""Benchmark 3 candidate pipeline with an explicitly restored Extractor."""

from agent_pipeline_v2_2.systems import _candidate_dependencies, build_candidate_system

from .extractor import extract_events


def build_system(llm, retriever, device: str = "auto"):
    dependencies = _candidate_dependencies()
    dependencies["extract"] = extract_events
    system = build_candidate_system(llm, retriever, device, dependencies=dependencies)
    system.extractor = extract_events
    return system
