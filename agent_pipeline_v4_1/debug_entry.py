"""Dispatch debug commands without changing the established 4.1 quality CLI."""

import sys


DEBUG_COMMANDS = {"validate-debug", "run-debug", "summary-debug"}


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] in DEBUG_COMMANDS:
        from .debug_cli import main as debug_main
        return debug_main(args)
    from .cli import main as quality_main
    return quality_main(args)
