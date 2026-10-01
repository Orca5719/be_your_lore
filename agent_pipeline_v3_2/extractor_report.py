from __future__ import annotations

import json
from pathlib import Path


def _pct(value):
    return "-" if value is None else f"{value:.2%}"


def render_extractor_audit(report: dict) -> str:
    total = report["totals"]
    lines = [
        "# Benchmark 3.2B — Extractor Failure Audit",
        "",
        "本报告只读取 Benchmark 3 trace；没有加载模型，也没有调用 generate。",
        "",
        f"- Stories: {report['stories']}",
        f"- Extractor calls: {report['extractor_calls']}",
        f"- Input tokens: {total['input_tokens']}",
        f"- Output tokens: {total['output_tokens']}",
        f"- Total LLM time: {total['total_time']:.3f} s",
        "",
        "## Cost classes",
        "",
        "| Class | Calls | Input tok | Output tok | Prefill s | Decode s | Total s | Share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in report["cost_classes"].items():
        lines.append(f"| {name} | {item['calls']} | {item['input_tokens']} | {item['output_tokens']} | {item['prefill_time']:.3f} | {item['decode_time']:.3f} | {item['total_time']:.3f} | {_pct(item['time_share'])} |")
    lines.extend([
        "",
        "`pure_waste` 是被完全丢弃的失败输出；成功 coverage recovery 单列为 `recovery_cost`；其余被采用输出为 `effective_cost`。",
        "",
        "## Failure taxonomy",
        "",
        "| Category | Calls | Stories | Output tok | Total s | Recovered later | Related schema |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    ranked = sorted(report["failure_categories"].items(), key=lambda pair: pair[1]["total_time"], reverse=True)
    for category, item in ranked:
        if item["calls"]:
            lines.append(f"| {category} | {item['calls']} | {item['stories']} | {item['output_tokens']} | {item['total_time']:.3f} | {item['recovered_calls']} | {item['related_schema']} |")
    lines.extend([
        "",
        "## Attempt outcomes",
        "",
        "| Outcome | Calls | Output tok | Total s |",
        "|---|---:|---:|---:|",
    ])
    for name, item in report["outcomes"].items():
        lines.append(f"| {name} | {item['calls']} | {item['output_tokens']} | {item['total_time']:.3f} |")
    lines.extend([
        "",
        "## Schema hypotheses ranked by measured waste",
        "",
        "| Rank | Category | Wasted s | Wasted output tok | Schema area | Hypothesis |",
        "|---:|---|---:|---:|---|---|",
    ])
    for item in report["recommendations"]:
        lines.append(f"| {item['rank']} | {item['category']} | {item['wasted_seconds']:.3f} | {item['wasted_output_tokens']} | {item['related_schema']} | {item['hypothesis']} |")
    closure = report["closure"]
    lines.extend([
        "",
        "## Accounting closure",
        "",
        f"- Calls closed: `{closure['calls_closed']}`",
        f"- Tokens closed: `{closure['tokens_closed']}`",
        f"- Time closed: `{closure['time_closed']}`",
        "",
        "这些根因是基于验证器错误和实际输出的结构归类；它们用于决定3.2C的schema实验，不代表模型能力的最终结论。",
    ])
    return "\n".join(lines) + "\n"


def write_extractor_markdown(output: Path, report: dict) -> Path:
    path = output / "extractor_failure_audit.md"
    path.write_text(render_extractor_audit(report), encoding="utf-8")
    return path
