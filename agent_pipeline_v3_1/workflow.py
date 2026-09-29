from __future__ import annotations

import json
from pathlib import Path

from .audit import _read_jsonl, audit_retries
from .generation import analyze_generation
from .report import write_generation_outputs
from .tokenizer import load_token_counter
from qwen_judge import MODEL, REVISION


def run_generation_audit(source: Path, output: Path) -> dict:
    source = Path(source).resolve()
    output = Path(output).resolve()
    retry_report = audit_retries(source, output)
    stories = _read_jsonl(source / "story_runs.jsonl")
    calls = _read_jsonl(output / "calls_audit.jsonl")
    token_counter = load_token_counter()
    analysis = analyze_generation(stories, calls, token_counter)
    analysis.update({
        "schema_version": "agent-pipeline-v3.1-generation-audit-v1",
        "source": str(source),
        "tokenizer": {"model": MODEL, "revision": REVISION, "local_files_only": True},
        "retry_audit": {
            "retry_calls": retry_report["retry_calls"],
            "totals": retry_report["totals"],
        },
        "observations": [
            "字段 token 来自锁定 Qwen tokenizer 的离线复算，不能与真实逐 token tracing 等同。",
            "精简投影保持事件、verdict 与冲突证据身份；时间收益按当前实测 decode 吞吐线性折算。",
            "该投影没有重新生成文本，因此只用于提出后续实验假设。",
        ],
    })
    call_rows = analysis.pop("call_rows")
    field_rows = analysis.pop("field_rows")
    projection_rows = analysis.pop("projection_rows")
    write_generation_outputs(output, analysis, call_rows, field_rows, projection_rows)
    return analysis


def regenerate_summary(directory: Path) -> Path:
    from .report import render_markdown

    directory = Path(directory).resolve()
    json_path = directory / "generation_audit.json"
    if not json_path.exists():
        raise ValueError("missing required file: generation_audit.json")
    report = json.loads(json_path.read_text(encoding="utf-8"))
    summary = directory / "generation_audit.md"
    summary.write_text(render_markdown(report), encoding="utf-8")
    return summary
