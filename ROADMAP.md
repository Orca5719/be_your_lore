# Benchmark 2.1 roadmap

Benchmark 2.1 is deliberately separated from the frozen Agent Pipeline v2 baseline.

1. **2.1 foundation** — establish the stricter Judge rule: a contradiction requires that the two propositions cannot both be true under the same scope.
2. **2.1A: Oracle Retrieval** — implemented and run on 72 reviewed cases; the committed result is a partial Oracle-Judge report, not an end-to-end retrieval benchmark.
3. **2.1B: Metadata-aware retrieval** — implemented with a versioned metadata audit; a full model comparison is still pending.
4. **2.1C: BM25 / hybrid RRF** — implemented with the six-configuration runner and story diagnostic; the complete retrieval matrix remains pending.

A planned branch is a persistent design marker, not evidence that its experiment has run. It will only be promoted to an executed benchmark after its implementation, data/fixture hashes, tests, and result report are committed.
