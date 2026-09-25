import numpy as np


def fixture():
    return {
        "items": [
            {"fixture_id": "f1", "fact": {"subject": "雷", "normalized_fact": "雷左侧心脏寄宿亚巴顿", "dimension": "physical_rule"}, "lore": [{"id": "a"}]},
            {"fixture_id": "f2", "fact": {"subject": "月城", "normalized_fact": "月城在2064年安装机械左臂", "dimension": "time"}, "lore": [{"id": "b"}, {"id": "c"}]},
        ]
    }


def chunks():
    return [
        {"id": "a", "text": "雷左侧心脏寄宿亚巴顿", "file": "x", "start_line": 1, "end_line": 1, "heading_path": ["雷"]},
        {"id": "b", "text": "月城在2064年安装机械左臂", "file": "x", "start_line": 2, "end_line": 2, "heading_path": ["月城"]},
        {"id": "c", "text": "2064年机械臂安装完成", "file": "x", "start_line": 3, "end_line": 3, "heading_path": ["2064年"]},
    ]


def metadata():
    return {
        "schema_version": "agent-pipeline-v2.1-lore-metadata-v1",
        "index_version": "idx",
        "entity_vocabulary": ["月城", "雷"],
        "items": [
            {"chunk_id": "a", "entities": ["雷"], "dimension_tags": ["physical_rule"], "time_markers": []},
            {"chunk_id": "b", "entities": ["月城"], "dimension_tags": ["time"], "time_markers": ["2064年"]},
            {"chunk_id": "c", "entities": [], "dimension_tags": ["time"], "time_markers": ["2064年"]},
        ],
    }


def test_benchmark_defines_exactly_six_ab_configs():
    from agent_pipeline_v2_1.retrieval_benchmark import CONFIGS

    assert [(row.method, row.metadata_filter) for row in CONFIGS] == [
        ("dense", False), ("dense", True),
        ("bm25", False), ("bm25", True),
        ("hybrid", False), ("hybrid", True),
    ]


def test_retrieval_metrics_distinguish_complete_set_and_micro_recall():
    from agent_pipeline_v2_1.retrieval_benchmark import score_rows

    rows = [
        {"expected_lore_ids": ["a"], "retrieved_ids": ["a", "x"], "candidate_ids": ["a", "x"]},
        {"expected_lore_ids": ["b", "c"], "retrieved_ids": ["b", "x"], "candidate_ids": ["b", "c", "x"]},
    ]
    metrics = score_rows(rows, k=2)

    assert metrics["complete_lore_recall_at_k"] == 0.5
    assert metrics["evidence_micro_recall_at_k"] == 2 / 3
    assert metrics["candidate_pool_complete_recall"] == 1.0
    assert metrics["mrr_at_k"] == 1.0


def test_run_configuration_returns_ranked_rows_and_metrics():
    from agent_pipeline_v2_1.contracts import RetrievalConfig
    from agent_pipeline_v2_1.retrieval_benchmark import run_configuration
    from agent_pipeline_v2_1.bm25 import BM25Retriever

    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.1, 0.995]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = run_configuration(RetrievalConfig("hybrid", False, 2), fixture(), chunks(), vectors, queries, BM25Retriever(chunks()), metadata())

    assert result["config_id"] == "hybrid__metadata_off"
    assert result["status"] == "ok"
    assert result["metrics"]["facts_total"] == 2
    assert len(result["rows"]) == 2


def test_markdown_contains_all_six_rows_and_required_metrics():
    from agent_pipeline_v2_1.retrieval_benchmark import render_retrieval_markdown

    results = []
    for method in ("dense", "bm25", "hybrid"):
        for enabled in (False, True):
            results.append({"config_id": f"{method}__metadata_{'on' if enabled else 'off'}", "method": method, "metadata_filter": enabled, "metrics": {"complete_lore_recall_at_k": 0.5, "evidence_micro_recall_at_k": 0.6, "mrr_at_k": 0.7, "candidate_pool_complete_recall": 0.8, "mean_candidate_count": 30.0, "ranking_seconds": 0.1}})
    text = render_retrieval_markdown({"top_k": 5, "results": results})
    assert text.count("| dense |") == 2
    assert "Complete Recall@5" in text
    assert "Candidate Pool Recall" in text


def test_repeat_aggregation_uses_metric_medians():
    from agent_pipeline_v2_1.retrieval_benchmark import aggregate_retrieval_runs

    runs = []
    for seconds in (3.0, 1.0, 2.0):
        runs.append({"query_encode_seconds": seconds / 2, "results": [{"config_id": "dense__metadata_off", "method": "dense", "metadata_filter": False, "status": "ok", "metrics": {"complete_lore_recall_at_k": 0.5, "ranking_seconds": seconds}, "rows": []}]})
    value = aggregate_retrieval_runs(runs, top_k=5)
    assert value["repeat_count"] == 3
    assert value["query_encode_seconds"] == 1.0
    assert value["results"][0]["metrics"]["ranking_seconds"] == 2.0


def test_retrieval_cli_defaults_lock_six_config_protocol():
    from agent_pipeline_v2_1.cli import build_parser

    args = build_parser().parse_args(["run-retrieval", "--device", "cuda"])
    assert args.top_k == 5
    assert args.repeats == 3
    assert args.query_batch_size == 16
