from __future__ import annotations

import csv
import json
from pathlib import Path


COMPONENT_COLUMNS = [
    "component", "calls", "retry_calls", "avg_batch_size", "input_tokens", "padded_input_tokens",
    "output_tokens", "prefill_seconds", "decode_seconds", "total_seconds", "ttft_p50_ms", "ttft_p95_ms",
    "prefill_useful_tokens_per_second", "prefill_compute_tokens_per_second", "decode_tokens_per_second",
    "peak_allocated", "peak_reserved", "llm_time_share",
]


def _number(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _gib(value):
    return "N/A" if value is None else f"{value / 2**30:.3f}"


def render_markdown(profile: dict, quality_guard: dict) -> str:
    workflow = profile["workflow"]
    lines = [
        "# Benchmark 3 LLM Workload Profile",
        "",
        f"Quality guard: {'PASS' if quality_guard.get('comparable') else 'FAIL'} "
        f"({quality_guard.get('matched_cases', 0)}/{quality_guard.get('total_reference_cases', 0)} cases matched)",
        "",
        "## Component profile",
        "",
        "| Component | Calls | Retry calls | Avg batch | Input tok | Padded tok | Output tok | Prefill s | Decode s | Total s | LLM time share | TTFT p50 ms | TTFT p95 ms | Prefill useful tok/s | Prefill compute tok/s | Decode tok/s | Peak allocated GiB | Peak reserved GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("extractor", "judge"):
        value = profile["components"][name]
        lines.append(
            "| " + ("Extractor" if name == "extractor" else "Judge") + " | "
            + " | ".join(
                [
                    str(value["calls"]), str(value["retry_calls"]), _number(value["avg_batch_size"]),
                    str(value["input_tokens"]), str(value["padded_input_tokens"]), str(value["output_tokens"]),
                    _number(value["prefill_seconds"]), _number(value["decode_seconds"]), _number(value["total_seconds"]),
                    "N/A" if value["llm_time_share"] is None else f"{value['llm_time_share']:.2%}",
                    _number(value["ttft_p50_ms"]), _number(value["ttft_p95_ms"]),
                    _number(value["prefill_useful_tokens_per_second"]), _number(value["prefill_compute_tokens_per_second"]),
                    _number(value["decode_tokens_per_second"]), _gib(value["peak_allocated"]), _gib(value["peak_reserved"]),
                ]
            ) + " |"
        )
    lines += [
        "",
        "## Workflow",
        "",
        "| Stage | Seconds | LLM calls |",
        "|---|---:|---:|",
    ]
    llm_calls = {
        "extraction": profile["components"]["extractor"]["calls"],
        "retrieval": 0,
        "judge": profile["components"]["judge"]["calls"],
        "report": workflow["report_llm_calls"],
    }
    for stage in ("extraction", "retrieval", "judge", "report"):
        lines.append(f"| {stage.title()} | {_number(workflow['stage_seconds'][stage])} | {llm_calls[stage]} |")
    lines += [
        f"| LLM generation | {_number(workflow['llm_generation_seconds'])} | {sum(llm_calls.values())} |",
        f"| Non-LLM | {_number(workflow['non_llm_seconds'])} | 0 |",
        f"| End-to-End | {_number(workflow['end_to_end_seconds'])} | {sum(llm_calls.values())} |",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Model load seconds | {_number(workflow['model_load_seconds'])} |",
        f"| Stories | {workflow['story_count']} |",
        f"| Facts | {workflow['fact_count']} |",
        f"| Stories/s | {_number(workflow['stories_per_second'], 5)} |",
        f"| Facts/s | {_number(workflow['facts_per_second'], 5)} |",
        f"| Calls/story | {_number(workflow['calls_per_story'])} |",
        f"| Tokens/story | {_number(workflow['tokens_per_story'])} |",
        "",
        "## Retry cost",
        "",
        "| Calls | Input tok | Output tok | Total seconds |",
        "|---:|---:|---:|---:|",
        f"| {profile['retry_cost']['calls']} | {profile['retry_cost']['input_tokens']} | {profile['retry_cost']['output_tokens']} | {_number(profile['retry_cost']['total_time'])} |",
        "",
        "## Slowest calls",
        "",
        "| Call | Story | Component | Batch | Input | Output | Total s |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for call in profile["slowest_calls"]:
        lines.append(
            f"| {call['call_id']} | {call.get('story_id') or ''} | {call['component']} | {call['batch_size']} | "
            f"{call['input_tokens']} | {call['output_tokens']} | {_number(call['total_time'])} |"
        )
    alignment = profile["alignment"]
    lines += [
        "",
        "## Timing alignment",
        "",
        "| Check | Seconds |",
        "|---|---:|",
        f"| Component total - call total | {_number(alignment.get('component_call_delta_seconds', alignment['component_llm_seconds'] - alignment['call_llm_seconds']), 6)} |",
        f"| Explained total - End-to-End | {_number(alignment.get('end_to_end_delta_seconds', alignment['explained_end_to_end_seconds'] - workflow['end_to_end_seconds']), 6)} |",
    ]
    lines += [
        "",
        "TTFT is measured after tokenization, from `model.generate()` start to the first completed generation step. "
        "Batch Judge TTFT is batch-level. VRAM peaks are per-call PyTorch process counters and are not summed.",
        "",
    ]
    return "\n".join(lines)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_profile_outputs(directory: Path, profile: dict, calls: list[dict], quality_guard: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    summary = {**profile, "quality_guard": quality_guard}
    (directory / "profile_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "profile_summary.md").write_text(render_markdown(profile, quality_guard), encoding="utf-8")
    component_rows = [{"component": name, **value} for name, value in profile["components"].items()]
    _write_csv(directory / "component_metrics.csv", COMPONENT_COLUMNS, component_rows)
    call_fields = sorted({key for row in calls for key in row}) if calls else ["call_id", "story_id", "component"]
    _write_csv(directory / "inference_calls.csv", call_fields, calls)
