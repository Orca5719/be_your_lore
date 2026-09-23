# Benchmark catalogue

| Branch | Status | Scope | Result / constraint |
|---|---|---|---|
| `benchmark/fact-level-v1` | frozen | 160 cases, 192 annotated facts | Historical reference; no fresh complete run after performance instrumentation. |
| `benchmark/story-level-pilot-v1` | frozen | 4-story pilot | Original manual reviews remain pending; not a final accuracy claim. |
| `benchmark/agent-pipeline-v1` | frozen | 4 stories, 5 draft facts | Scored provisional baseline. |
| `benchmark/agent-pipeline-v2` | executed | 24 stories, 72 gold facts; 48 Judge fixtures | Draft annotation and assistant-reviewed ledger. Story conflict F1: 0.4889; Oracle-Judge accuracy: 0.7917. |
| `benchmark/2.1-foundation` | partial implementation | Judge coexistence contract and scope guard | No retrieval experiment or 2.1 model benchmark has run. |
| `benchmark/2.1a-oracle-retrieval` | reserved | Oracle-retrieval ablation | Planned only. |
| `benchmark/2.1b-metadata-retrieval` | reserved | Metadata-aware dense retrieval ablation | Planned only. |
| `benchmark/2.1c-bm25-hybrid-rrf` | reserved | BM25 and Dense+BM25/RRF ablation | Planned only. |

Each branch contains a `STATUS.md` that states its exact evidence boundary and whether it may be interpreted as a completed benchmark.
