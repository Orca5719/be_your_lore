"""Small independent CLI surface for Agent Pipeline 2.1."""

from __future__ import annotations

import argparse
import json

from . import VERSION
from .contracts import BenchmarkConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent_pipeline_v2_1_benchmark.py")
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser("info", help="显示2.1版本与实施状态")
    info.add_argument("--json", action="store_true", dest="as_json")
    return parser


def benchmark_info() -> dict:
    config = BenchmarkConfig()
    return {
        "schema_version": "agent-pipeline-v2.1-benchmark-info-v1",
        "package": "agent_pipeline_v2_1",
        "version": VERSION,
        "judge_batch_size": config.judge_batch_size,
        "repeats": config.repeats,
        "implemented_stages": ["scaffold", "judge_contract"],
        "planned_benchmarks": ["2.1A", "2.1B", "2.1C"],
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "info":
        value = benchmark_info()
        print(json.dumps(value, ensure_ascii=False, indent=2) if args.as_json else "Agent Pipeline 2.1 scaffold ready")
        return 0
    raise ValueError("未知命令")
