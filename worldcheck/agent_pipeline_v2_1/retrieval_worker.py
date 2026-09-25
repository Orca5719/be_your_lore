"""In-process runner for the six Benchmark 2.1B retrieval configurations."""

from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np

from index_store import validate as validate_index

from .bm25 import BM25Retriever
from .contracts import RetrievalConfig
from .retrieval_benchmark import aggregate_retrieval_runs, run_configuration


def run_retrieval_worker(root: Path, fixture_path: Path, metadata_path: Path, device: str, top_k: int, repeats: int, query_batch_size: int) -> dict:
    if top_k < 1 or repeats < 1 or query_batch_size < 1:
        raise ValueError("top-k、repeats和query-batch-size必须为正整数")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8-sig"))
    lore_metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    version = fixture["index_version"]
    version_dir = root / "data" / "index" / "versions" / version
    index_metadata = json.loads((version_dir / "metadata.json").read_text(encoding="utf-8-sig"))
    vectors = np.load(version_dir / "embeddings.npy", allow_pickle=False)
    validate_index(vectors, index_metadata)
    if lore_metadata.get("index_version") != version:
        raise ValueError("lore metadata与fixture索引版本不匹配")
    from encoder import Encoder, MODEL, REVISION

    load_started = time.perf_counter()
    encoder = Encoder(device=device, offline=True, precision="float32")
    model_load_seconds = time.perf_counter() - load_started
    torch = __import__("torch")
    actual_device = encoder.device
    queries = [item["fact"]["normalized_fact"] for item in fixture["items"]]
    encoder.encode_queries(queries[: min(query_batch_size, len(queries))], batch_size=query_batch_size)
    if actual_device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    bm25 = BM25Retriever(index_metadata["chunks"])
    runs = []
    for repeat in range(1, repeats + 1):
        if actual_device == "cuda":
            torch.cuda.synchronize()
        encode_started = time.perf_counter()
        query_vectors = encoder.encode_queries(queries, batch_size=query_batch_size)
        if actual_device == "cuda":
            torch.cuda.synchronize()
        encode_seconds = time.perf_counter() - encode_started
        print(f"  retrieval repeat={repeat}/{repeats} query vectors ready", flush=True)
        results = []
        for method in ("dense", "bm25", "hybrid"):
            for enabled in (False, True):
                config = RetrievalConfig(method, enabled, top_k)
                result = run_configuration(config, fixture, index_metadata["chunks"], vectors, query_vectors, bm25, lore_metadata)
                result["metrics"]["query_encode_seconds"] = encode_seconds if method in {"dense", "hybrid"} else 0.0
                result["metrics"]["total_retrieval_seconds"] = result["metrics"]["ranking_seconds"] + result["metrics"]["query_encode_seconds"]
                result["metrics"]["latency_per_fact_seconds"] = result["metrics"]["total_retrieval_seconds"] / len(queries)
                results.append(result)
                print(f"    {result['config_id']} complete_recall={result['metrics']['complete_lore_recall_at_k']:.4f}", flush=True)
        runs.append({"repeat": repeat, "query_encode_seconds": encode_seconds, "results": results})
    report = aggregate_retrieval_runs(runs, top_k)
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30 if actual_device == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30 if actual_device == "cuda" else None
    report.update({
        "device": actual_device,
        "model": MODEL,
        "revision": REVISION,
        "precision": "FP32",
        "model_load_seconds": model_load_seconds,
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "query_batch_size": query_batch_size,
        "index_version": version,
        "runs": runs,
    })
    return report

