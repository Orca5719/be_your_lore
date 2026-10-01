from __future__ import annotations

import csv
import json
from pathlib import Path


def _pct(value):
    return "-" if value is None else f"{value * 100:.2f}%"


def _num(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def write_outputs(directory: Path, summary: dict, comparisons: list[dict], calls: list[dict]) -> None:
    (directory / "judge_lean_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (directory / "judge_calls.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["call_id", "story_id", "status", "batch_size", "input_tokens", "padded_input_tokens", "output_tokens", "prefill_time", "decode_time", "total_time", "ttft_ms", "peak_allocated", "peak_reserved", "is_retry", "request_ids"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in calls:
            writer.writerow({**row, "request_ids": "|".join(row.get("request_ids", []))})
    with (directory / "verdict_changes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["case_id", "event_id", "old_verdict", "new_verdict", "changed", "evidence_ids", "chunk_ids"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    quality = summary["quality"]
    perf = summary["performance"]
    old = summary["baseline_judge"]
    new = perf["components"]["judge"]
    delta = summary["delta"]
    lines = [
        "# Benchmark 3.2A — Lean Judge",
        "",
        "冻结 Benchmark 3 的 Extraction 与 Retrieval，只重新运行 Judge。Lean Judge 只生成 verdict 与 evidence_ids。",
        "",
        "## Quality",
        "",
        "| Metric | Benchmark 3 | Lean Judge |",
        "|---|---:|---:|",
        f"| Judge Accuracy | {_pct(summary['baseline_quality']['judge']['accuracy'])} | {_pct(quality['judge']['accuracy'])} |",
        f"| Judge FP | {summary['baseline_quality']['judge']['fp']} | {quality['judge']['fp']} |",
        f"| Judge FN | {summary['baseline_quality']['judge']['fn_total']} | {quality['judge']['fn_total']} |",
        f"| End-to-End Precision | {_pct(summary['baseline_quality']['end_to_end_conflict']['precision'])} | {_pct(quality['end_to_end_conflict']['precision'])} |",
        f"| End-to-End Recall | {_pct(summary['baseline_quality']['end_to_end_conflict']['recall'])} | {_pct(quality['end_to_end_conflict']['recall'])} |",
        f"| End-to-End F1 | {_pct(summary['baseline_quality']['end_to_end_conflict']['f1'])} | {_pct(quality['end_to_end_conflict']['f1'])} |",
        "",
        "## Judge performance",
        "",
        "| Metric | Benchmark 3 | Lean Judge | Change |",
        "|---|---:|---:|---:|",
        f"| Calls | {old['calls']} | {new['calls']} | {new['calls'] - old['calls']} |",
        f"| Retry calls | {old['retry_calls']} | {new['retry_calls']} | {new['retry_calls'] - old['retry_calls']} |",
        f"| Output tokens | {old['output_tokens']} | {new['output_tokens']} | {_pct(delta['output_token_change_ratio'])} |",
        f"| Prefill time (s) | {_num(old['prefill_seconds'])} | {_num(new['prefill_seconds'])} | {_pct(delta['prefill_change_ratio'])} |",
        f"| Decode time (s) | {_num(old['decode_seconds'])} | {_num(new['decode_seconds'])} | {_pct(delta['decode_change_ratio'])} |",
        f"| LLM total time (s) | {_num(old['total_seconds'])} | {_num(new['total_seconds'])} | {_pct(delta['total_change_ratio'])} |",
        f"| TTFT p50 (ms) | {_num(old['ttft_p50_ms'])} | {_num(new['ttft_p50_ms'])} | - |",
        f"| TTFT p95 (ms) | {_num(old['ttft_p95_ms'])} | {_num(new['ttft_p95_ms'])} | - |",
        f"| Decode tok/s | {_num(old['decode_tokens_per_second'])} | {_num(new['decode_tokens_per_second'])} | - |",
        f"| Peak allocated (bytes) | {old['peak_allocated']} | {new['peak_allocated']} | - |",
        f"| Peak reserved (bytes) | {old['peak_reserved']} | {new['peak_reserved']} | - |",
        "",
        f"Verdict changed: {sum(row['changed'] for row in comparisons)}/{len(comparisons)} facts.",
        "",
        "本实验没有自然语言解释，也不使用旧 assessment scope guard；质量指标直接反映 Lean Judge 的分类能力。",
    ]
    (directory / "judge_lean_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
