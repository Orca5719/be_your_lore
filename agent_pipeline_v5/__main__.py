"""Offline-only CLI."""
import argparse
from datetime import datetime
from pathlib import Path
from .diagnostics import audit_archives


def main():
    parser = argparse.ArgumentParser(description='Benchmark 5A offline audit; no model loaded')
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--source-root', type=Path, required=True)
    audit.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or Path(__file__).parent / 'reports' / ('audit_5A_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    try:
        audit_archives(args.source_root, output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f'错误：{exc}\n')
    print('SUMMARY=' + str((output / 'audit_summary.md').resolve()))


if __name__ == '__main__':
    main()
