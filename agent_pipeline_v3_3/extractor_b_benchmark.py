"""Extractor-only Benchmark 3.3B runner and comparison with the 3.3 base run."""

from __future__ import annotations

import json
from pathlib import Path
import time

from agent_pipeline_v2_1.extractor import ExtractionRepairLLM
from agent_pipeline_v3.metrics import aggregate_profile
from agent_pipeline_v3_2.lean_extractor_benchmark import (
    _atomic_jsonl, build_extraction_row, candidate_review,
    reuse_exact_event_reviews, score_extraction_case_aware,
)

from .extractor_b import extract_events_b


def run_cases(cases, llm, output: Path, progress=None):
    path = output / "extractor_runs.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    expected_ids = [case["id"] for case in cases]
    found_ids = [row.get("case_id") for row in rows]
    if found_ids != expected_ids[:len(found_ids)]:
        raise ValueError("已有提取结果不是数据集的连续前缀，拒绝续跑")
    llm.collector.resume_after([call for row in rows for call in row.get("calls", [])])
    repaired = ExtractionRepairLLM(llm)
    for case in cases[len(rows):]:
        llm.set_story_id(case["id"])
        before = len(llm.collector.records)
        started = time.perf_counter()
        extraction = extract_events_b(case["story"], device=llm.device, llm=repaired)
        elapsed = time.perf_counter() - started
        row = build_extraction_row(case["id"], extraction, llm.collector.records[before:], elapsed)
        row["system"] = "3.3B-deterministic-coverage"
        rows.append(row)
        _atomic_jsonl(path, rows)
        if progress:
            progress(len(rows), len(cases), case["id"])
    return rows


def summarize_attempts(rows: list[dict], calls: list[dict]) -> dict:
    by_id = {call["call_id"]: call for call in calls}
    groups = {name: [] for name in ("accepted_partial", "superseded_partial", "failed_generation", "local_recovery", "full_fallback")}
    full_fallback_adopted = 0
    uncovered = 0
    for row in rows:
        extraction = row["result"]["benchmark_stages"]["extraction"]
        uncovered += len(extraction.get("uncovered_span_ids", []))
        for window in extraction.get("calls", []):
            for attempt in window.get("attempts", []):
                call_id = attempt.get("timing", {}).get("call_id")
                if call_id not in by_id:
                    raise ValueError("提取尝试缺少对应的LLM调用：" + str(call_id))
                if attempt.get("purpose") == "recover_missing_targets":
                    groups["local_recovery"].append(by_id[call_id])
                elif attempt.get("purpose") == "fallback_full_window":
                    groups["full_fallback"].append(by_id[call_id])
                    full_fallback_adopted += bool(attempt.get("adopted"))
                elif attempt.get("status") == "error":
                    groups["failed_generation"].append(by_id[call_id])
                elif attempt.get("coverage_pending"):
                    group = "accepted_partial" if attempt.get("adopted", True) else "superseded_partial"
                    groups[group].append(by_id[call_id])
    def total(name, field):
        return sum(call[field] for call in groups[name])
    return {
        "accepted_partial_calls": len(groups["accepted_partial"]),
        "accepted_partial_seconds": total("accepted_partial", "total_time"),
        "accepted_partial_output_tokens": total("accepted_partial", "output_tokens"),
        "superseded_partial_calls": len(groups["superseded_partial"]),
        "superseded_partial_seconds": total("superseded_partial", "total_time"),
        "superseded_partial_output_tokens": total("superseded_partial", "output_tokens"),
        "failed_generation_calls": len(groups["failed_generation"]),
        "failed_generation_seconds": total("failed_generation", "total_time"),
        "local_recovery_calls": len(groups["local_recovery"]),
        "local_recovery_seconds": total("local_recovery", "total_time"),
        "full_fallback_calls": len(groups["full_fallback"]),
        "full_fallback_adopted": full_fallback_adopted,
        "full_fallback_seconds": total("full_fallback", "total_time"),
        "uncovered_span_count": uncovered,
    }


def summarize(dataset, source_rows, source_profile, candidate_rows, review, model_load_seconds):
    calls = [call for row in candidate_rows for call in row.get("calls", [])]
    profile = aggregate_profile(calls, candidate_rows, model_load_seconds)
    ledger = reuse_exact_event_reviews(dataset, source_rows, candidate_rows, review)
    baseline_quality = score_extraction_case_aware(dataset["cases"], source_rows, candidate_review(review))
    candidate_quality = score_extraction_case_aware(dataset["cases"], candidate_rows, ledger)
    attempts = summarize_attempts(candidate_rows, calls)
    formal = ledger["pending_event_count"] == 0 and len(candidate_rows) == len(dataset["cases"])
    baseline = source_profile["components"]["extractor"]
    current = profile["components"]["extractor"]
    quality_gate = "pending_review" if not formal else (
        "pass" if candidate_quality["recall"]["hits"] >= baseline_quality["recall"]["hits"] - 1
        and candidate_quality["hallucination_rate"]["count"] <= baseline_quality["hallucination_rate"]["count"]
        and attempts["uncovered_span_count"] == 0 else "fail"
    )
    return {
        "schema_version": "agent-pipeline-v3.3B-extractor-summary-v1",
        "stories": len(candidate_rows),
        "events": sum(len(row["result"]["benchmark_stages"]["extraction"]["events"]) for row in candidate_rows),
        "review": ledger, "quality_formal": formal, "quality_gate": quality_gate,
        "quality": {"base": baseline_quality, "candidate": candidate_quality},
        "performance": {"base": baseline, "candidate": current, "profile": profile},
        "attempts": attempts,
    }


def render_report(summary: dict) -> str:
    old = summary["performance"]["base"]
    new = summary["performance"]["candidate"]
    attempts = summary["attempts"]
    quality = summary["quality"]
    lines = [
        "# Benchmark 3.3B — Deterministic Coverage", "",
        "原版 Extractor 提示、窗口、事件字段和生成上限不变；只比较程序侧覆盖账本与局部补提策略。Judge 不参与本实验。",
        f"24篇完成数：{summary['stories']}；提取事件：{summary['events']}；剩余未覆盖span：{attempts['uncovered_span_count']}。",
        f"质量状态：**{summary['quality_gate']}**；待人工核对的新事件：{summary['review']['pending_event_count']}。",
        "", "## Performance", "",
        "| Metric | 3.3 Base | 3.3B |", "|---|---:|---:|",
        f"| Extractor calls | {old['calls']} | {new['calls']} |",
        f"| Retry calls | {old['retry_calls']} | {new['retry_calls']} |",
        f"| Input tokens | {old['input_tokens']} | {new['input_tokens']} |",
        f"| Output tokens | {old['output_tokens']} | {new['output_tokens']} |",
        f"| Prefill s | {old['prefill_seconds']:.3f} | {new['prefill_seconds']:.3f} |",
        f"| Decode s | {old['decode_seconds']:.3f} | {new['decode_seconds']:.3f} |",
        f"| LLM total s | {old['total_seconds']:.3f} | {new['total_seconds']:.3f} |",
        f"| TTFT p50 ms | {old['ttft_p50_ms']:.1f} | {new['ttft_p50_ms']:.1f} |",
        f"| Peak allocated bytes | {old['peak_allocated']} | {new['peak_allocated']} |",
        "", "## Coverage calls", "",
        f"- 首轮有效但遗漏部分span：{attempts['accepted_partial_calls']}次，{attempts['accepted_partial_seconds']:.3f}秒，{attempts['accepted_partial_output_tokens']}输出tokens；内容被保留，不计为纯浪费。",
        f"- 后来被整窗兜底替换的首轮输出：{attempts['superseded_partial_calls']}次，{attempts['superseded_partial_seconds']:.3f}秒，{attempts['superseded_partial_output_tokens']}输出tokens；单列为被替换成本。",
        f"- 局部补提：{attempts['local_recovery_calls']}次，{attempts['local_recovery_seconds']:.3f}秒。",
        f"- 局部补提未完成后的整窗兜底：{attempts['full_fallback_calls']}次，采纳{attempts['full_fallback_adopted']}次，{attempts['full_fallback_seconds']:.3f}秒。",
        f"- 格式或字段失败的调用：{attempts['failed_generation_calls']}次，{attempts['failed_generation_seconds']:.3f}秒。",
        "", "## Extraction quality", "",
        "| Metric | 3.3 Base | 3.3B |", "|---|---:|---:|",
    ]
    for name, key in (("Gold hits", "recall"), ("Valid events", "precision")):
        b, c = quality["base"][key], quality["candidate"][key]
        lines.append(f"| {name} | {b['hits']}/{b['total']} | {c['hits']}/{c['total']} |")
    b = quality["base"]["hallucination_rate"]["count"]
    c = quality["candidate"]["hallucination_rate"]["count"]
    lines.append(f"| Unsupported events | {b} | {c} |")
    lines.extend(["", "如有待核对事件，3.3B质量数字只是已继承人工标签的下界，不可当作正式召回率。",
                  "这是独立单轮运行；耗时差异同时受GPU运行状态影响，不能把全部差值归因于覆盖策略。", ""])
    return "\n".join(lines)
