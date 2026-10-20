import sys

if len(sys.argv) > 1 and sys.argv[1] in {"validate-continuous", "run-continuous", "summary-continuous"}:
    from .cli_continuous import main
else:
    from .cli import main

raise SystemExit(main())
