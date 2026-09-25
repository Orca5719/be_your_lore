# Benchmark 2.1B — Metadata-Aware Retrieval

**State:** implemented; structural audit complete, no full model benchmark run.

## Question

Can structured source metadata remove irrelevant lore before retrieval without excluding evidence required to judge a fact?

## Included material

- Versioned lore metadata snapshot and metadata-filter implementation.
- Audit over the 72 Oracle cases: the filtered candidate pool preserves required lore for 67 cases (93.06%).
- Dedicated metadata and filter tests.

This branch does not claim a completed model or timing comparison. It is the retrieval-filter layer that precedes 2.1C's BM25/RRF experiments.
