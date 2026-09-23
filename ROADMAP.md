# Benchmark 2.1 roadmap

Benchmark 2.1 is deliberately separated from the frozen Agent Pipeline v2 baseline.

1. **2.1 foundation** — establish the stricter Judge rule: a contradiction requires that the two propositions cannot both be true under the same scope.
2. **2.1A: Oracle Retrieval** — isolate Judge behavior by feeding the required evidence directly.
3. **2.1B: Metadata-aware retrieval** — measure the effect of metadata filtering while holding the dense retriever and dataset constant.
4. **2.1C: BM25 / hybrid RRF** — compare lexical, dense, and fused retrieval under the same evaluation protocol.

A planned branch is a persistent design marker, not evidence that its experiment has run. It will only be promoted to an executed benchmark after its implementation, data/fixture hashes, tests, and result report are committed.
