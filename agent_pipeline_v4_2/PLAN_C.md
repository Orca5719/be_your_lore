# 4.2C implementation plan

Goal: compare the frozen Extractor under one vLLM backend with batch limits 1/2/4 and n-gram lengths 0/3/5. User authorized implementation on 2026-10-18; long GPU runs remain user-operated in WSL2.

Architecture: reuse the frozen extraction and repair functions and StaticBatcher; feed independent tokenized requests to vLLM offline batched generation. This is bounded batch dispatch, not cross-batch continuous admission. Each arm runs in a fresh subprocess. Export the existing pinned HF model as prequantized NF4/double-quant/BF16 once; do not silently re-quantize in vLLM. Keep A/B untouched.

Constraints: no GPU benchmark during development; no OS/WSL installation; no automatic weight download; preserve model revision, prompt, schema, token budget, retries. New backend numeric equivalence is not assumed. Screen runs never publish full quality. Fresh full arms only have comparable wall times; resumed partial arms invalidate wall comparison. Audit every event afresh to avoid reusing known incorrect historical labels.

- [x] Add tests for model artifact integrity, batch request mapping, length failures, configuration matrix, resume and audit gates.
- [x] Implement `vllm_model.py`: export/validate artifacts, pinned engine configuration, tokenized batching and truthful metrics.
- [x] Implement `vllm_benchmark.py`: preflight, screening/full matrix, per-arm subprocess isolation, checkpoints, summaries and frozen downstream reuse.
- [x] Document WSL setup and commands in README; validate CLI and run focused and full tests.

Review focus: missing Linux dependencies fail before work; no old result overwrite; missing profiling metrics remain unknown; screening cannot become a full result on resume; partial output cannot be reported as quality-approved; model/config/runtime changes invalidate resume.

## Execution ledger

- CodeGraph consulted; its index surfaced older modules rather than new 4.2 files. Read the known 4.2 entrypoints and frozen adapters directly. Reused existing clean benchmark-4.2 worktree; main checkout left untouched.
- Ruling: use vLLM offline batched generation through existing StaticBatcher, explicitly label bounded batch dispatch. No custom continuous scheduler is introduced.
- Ruling: use prequantized export and fixed EOS list; do not silently substitute inflight quantization or default stopping rules.
- RED: new module import missing; implemented adapter/CLI. Additional RED→GREEN checks caught JSON tuple/list manifest mismatch, missing second EOS, incomplete extraction approval, and asynchronous scheduling defaults.
- Final independent read-only review: fixed incomplete extraction gate, downstream source/prompt identity, and async_scheduling=False across all arms. Unmeasured GPU/backend compatibility remains a user-run prerequisite, not a success claim.
- Verification: 26 focused A/B/C tests passed (13 new C tests). Full suite: 637 passed, 2 failed because ignored historical 3.3 reports are absent in this worktree; see VALIDATION_C.md. CLI help and Windows preflight refusal verified; git diff --check passed. No real model export, Linux install or GPU benchmark executed.
