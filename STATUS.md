# Benchmark 2.1A — Oracle Retrieval

**State:** reserved; not implemented and not run.

## Intended question

How accurately does Judge 2.1 classify the benchmark facts when it receives the required lore evidence directly, without retrieval error?

## Planned controls

- Hold the Judge 2.1 prompt, model, fixture set, batch size and repeat count fixed.
- Supply each case's annotated minimum evidence set as the retrieval context.
- Report three-class accuracy, macro-F1, per-class metrics, parsing/retry failures and batch performance.
- Record fixture, prompt, lore and model identifiers before comparing with 2.1B or 2.1C.

No code, fixture, result, or numerical claim belongs to this branch yet.
