# Benchmark 2.1 foundation

**State:** partially implemented; no 2.1 retrieval benchmark has run.

This branch starts from the archived Agent Pipeline v2 material and adds the isolated `agent_pipeline_v2_1` package. It establishes the Judge 2.1 coexistence rule: a contradiction requires proof that the two propositions cannot both be true under the same subject, time, place, conditions and scope. The deterministic scope guard downgrades unsupported explicit conclusions to `uncertain` while preserving the model verdict for analysis.

The version scaffold exposes intended retrieval modes (`dense`, `bm25`, `hybrid`) and metadata filtering as configuration contracts only. Oracle Retrieval, metadata-aware retrieval, BM25 and RRF are not implemented here; the three dedicated successor branches are design markers rather than results.
