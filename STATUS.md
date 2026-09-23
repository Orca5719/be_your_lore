# Benchmark 2.1B — Metadata-aware retrieval

**State:** reserved; not implemented and not run.

## Intended question

Does filtering dense-retrieval candidates by compatible metadata improve evidence quality and downstream Judge 2.1 decisions relative to the unfiltered dense baseline?

## Planned controls

- Keep the corpus, dense encoder, query formation, top-K, Judge protocol and dataset fixed.
- Compare metadata filtering on/off using the same cases and gold evidence requirements.
- Report retrieval Recall@K, MRR, Precision@K, noise, missing retrievals and downstream classification/detection outcomes.
- Preserve a per-case retrieval trace and configuration digest.

No code, result, or numerical claim belongs to this branch yet.
