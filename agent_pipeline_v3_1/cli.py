from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from .audit import audit_retries, validate_result


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "agent_pipeline_v3_1" / "reports"


def _default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPORT_ROOT / f"retry_audit_{stamp}_{uuid.uuid4().hex[:6]}"


def validate_command(args) -> int:
    print(json.dumps(validate_result(args.result_dir.resolve()), ensure_ascii=False, indent=2))
    return 0


def audit_command(args) -> int:
    output = (args.output or _default_output()).resolve()
    report = audit_retries(args.result_dir.resolve(), output)
    print("AUDIT_DIR=" + str(output), flush=True)
    print(json.dumps({"retry_calls": report["retry_calls"], "totals": report["totals"]}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark 3.1 read-only generation retry audit")
    commands = parser.add_subparsers(dest="command", required=True)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--result-dir", type=Path, required=True)
    validate_parser.set_defaults(handler=validate_command)
    audit_parser = commands.add_parser("audit-retries")
    audit_parser.add_argument("--result-dir", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path)
    audit_parser.set_defaults(handler=audit_command)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print("错误：" + str(exc), flush=True)
        return 2
