# Benchmark 2.1C — BM25 / hybrid RRF retrieval

**State:** reserved; not implemented and not run.

## Intended question

How do lexical BM25, dense retrieval, and their reciprocal-rank fusion compare under the same 2.1 data and Judge protocol?

## Planned controls

- Use the same corpus, metadata policy, query inputs, top-K and gold evidence definitions for every retrieval mode.
- Evaluate `bm25`, `dense`, and `hybrid` modes separately; record RRF configuration for hybrid runs.
- Report retrieval quality/noise and all downstream Judge and finding-level detection metrics.
- Do not compare performance or accuracy across runs whose corpus, fixture, prompt or model digest differs.

No code, result, or numerical claim belongs to this branch yet.
