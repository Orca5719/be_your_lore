from __future__ import annotations

import csv
import json
from pathlib import Path


def render_markdown(report: dict) -> str:
    lines = [
        "# Benchmark 3.1 Generation Audit",
        "",
        f"- 故事：{report.get('stories', 0)}",
        f"- 最终事件：{report.get('facts', 0)}",
        "- 实测列来自 Benchmark 3 trace；字段 token 是 tokenizer 离线复算的估算。",
        "- **Counterfactual estimate** 只回答‘若输出采用精简结构，按当前吞吐理论上可少解码多少’；不代表真实优化结果。",
        "",
        "## 生成成本",
        "",
        "| Component | Calls | 实测 output tokens | 实测 decode s | Projected tokens | Reduction | Projected decode s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in report.get("components", {}).items():
        projected_time = item.get("projected_decode_time")
        lines.append(
            f"| {name} | {item.get('calls', 0)} | {item.get('actual_output_tokens', 0)} | "
            f"{item.get('actual_decode_time', 0):.3f} | {item.get('projected_output_tokens', 0)} | "
            f"{item.get('reduction_ratio', 0):.1%} | "
            f"{projected_time:.3f} |" if projected_time is not None else
            f"| {name} | {item.get('calls', 0)} | {item.get('actual_output_tokens', 0)} | "
            f"{item.get('actual_decode_time', 0):.3f} | {item.get('projected_output_tokens', 0)} | "
            f"{item.get('reduction_ratio', 0):.1%} | n/a |"
        )
    lines.extend(["", "## 字段成本（tokenizer 离线估算）", "", "| Component | Field | Estimated tokens |", "|---|---|---:|"])
    for name, item in report.get("components", {}).items():
        for field, value in item.get("field_estimated_tokens", {}).items():
            lines.append(f"| {name} | {field} | {value} |")
    lines.extend(["", "## Extractor 输出类别（实测 token）", "", "| Class | Calls | Output tokens |", "|---|---:|---:|"])
    for name, value in report.get("components", {}).get("extractor", {}).get("output_classes", {}).items():
        lines.append(f"| {name} | {value.get('calls', 0)} | {value.get('actual_output_tokens', 0)} |")
    lines.extend(["", "## Judge verdict 分布", "", "| Verdict | Rows | Estimated field tokens |", "|---|---:|---:|"])
    for verdict, value in report.get("components", {}).get("judge", {}).get("verdicts", {}).items():
        lines.append(f"| {verdict} | {value.get('rows', 0)} | {value.get('estimated_tokens', 0)} |")
    retry = report.get("retry_audit", {})
    if retry:
        totals = retry.get("totals", {})
        lines.extend([
            "", "## Retry 实测成本", "",
            f"- 调用：{retry.get('retry_calls', 0)}",
            f"- 输出 token：{totals.get('output_tokens', 0)}",
            f"- 总耗时：{totals.get('total_time', 0):.3f} 秒",
        ])
    lines.extend(["", "## 观察", ""])
    for observation in report.get("observations", []):
        lines.append(f"- {observation}")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_generation_outputs(
    directory: Path,
    report: dict,
    calls: list[dict],
    field_rows: list[dict],
    projection_rows: list[dict],
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "generation_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "generation_audit.md").write_text(render_markdown(report), encoding="utf-8")
    _write_csv(directory / "calls_audit.csv", calls)
    _write_csv(directory / "field_token_breakdown.csv", field_rows)
    _write_csv(directory / "counterfactual_projection.csv", projection_rows)
    retry_path = directory / "retry_taxonomy.csv"
    if not retry_path.exists():
        _write_csv(retry_path, report.get("retry_rows", []))
