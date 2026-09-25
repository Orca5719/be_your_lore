def dense():
    return [
        {"id": "a", "text": "A", "score": 0.91},
        {"id": "b", "text": "B", "score": 0.89},
        {"id": "c", "text": "C", "score": 0.50},
    ]


def lexical():
    return [
        {"id": "b", "text": "B", "score": 100.0, "lexical_score": 100.0},
        {"id": "d", "text": "D", "score": 90.0, "lexical_score": 90.0},
        {"id": "a", "text": "A", "score": 1.0, "lexical_score": 1.0},
    ]


def test_rrf_rewards_items_supported_by_both_rankers():
    from agent_pipeline_v2_1.rrf import fuse_rrf

    report = fuse_rrf(dense(), lexical(), k=4, rrf_k=60)

    assert report["method"] == "dense+bm25-rrf"
    assert [row["id"] for row in report["results"]][:2] == ["b", "a"]
    assert report["results"][0]["dense_rank"] == 2
    assert report["results"][0]["bm25_rank"] == 1
    assert report["results"][0]["dense_score"] == 0.89
    assert report["results"][0]["bm25_score"] == 100.0


def test_rrf_uses_ranks_not_incompatible_raw_score_scales():
    from agent_pipeline_v2_1.rrf import fuse_rrf

    first = fuse_rrf(dense(), lexical(), k=4)
    changed = [dict(row, score=row["score"] * 1_000_000, lexical_score=row["lexical_score"] * 1_000_000) for row in lexical()]
    second = fuse_rrf(dense(), changed, k=4)

    assert [row["id"] for row in first["results"]] == [row["id"] for row in second["results"]]
    assert [row["rrf_score"] for row in first["results"]] == [row["rrf_score"] for row in second["results"]]


def test_rrf_weights_are_explicit_and_can_change_priority():
    from agent_pipeline_v2_1.rrf import fuse_rrf

    result = fuse_rrf(dense(), lexical(), k=1, dense_weight=4.0, bm25_weight=1.0)
    assert result["results"][0]["id"] == "a"
    assert result["config"] == {"rrf_k": 60, "dense_weight": 4.0, "bm25_weight": 1.0}


def test_rrf_union_keeps_single_ranker_items_and_marks_missing_rank():
    from agent_pipeline_v2_1.rrf import fuse_rrf

    rows = fuse_rrf(dense(), lexical(), k=10)["results"]
    by_id = {row["id"]: row for row in rows}
    assert by_id["c"]["bm25_rank"] is None
    assert by_id["d"]["dense_rank"] is None
    assert by_id["d"]["sources"] == ["bm25"]


def test_rrf_rejects_duplicate_ids_and_invalid_parameters():
    from agent_pipeline_v2_1.rrf import fuse_rrf

    bad = [dense()[0], dense()[0]]
    for kwargs in ({"dense_results": bad, "bm25_results": lexical()}, {"dense_results": dense(), "bm25_results": lexical(), "rrf_k": 0}, {"dense_results": dense(), "bm25_results": lexical(), "dense_weight": 0}):
        try:
            fuse_rrf(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("非法RRF输入必须失败")


def test_unified_retriever_exposes_dense_bm25_and_hybrid_with_same_contract():
    import numpy as np
    from agent_pipeline_v2_1.retrieval import V21Retriever

    class Encoder:
        device = "cpu"

        def encode_queries(self, queries):
            return np.asarray([[1.0, 0.0] for _ in queries], dtype=np.float32)

    class Dense:
        encoder = Encoder()
        vectors = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
        metadata = {"chunks": [
            {"id": "a", "text": "雷的左侧心脏寄宿亚巴顿", "file": "x", "start_line": 1, "end_line": 1, "heading_path": ["雷"]},
            {"id": "b", "text": "雷的右侧心脏寄宿拉古艾尔", "file": "x", "start_line": 2, "end_line": 2, "heading_path": ["雷"]},
            {"id": "c", "text": "大地震发生", "file": "x", "start_line": 3, "end_line": 3, "heading_path": ["历史"]},
        ]}

    meta = {
        "schema_version": "agent-pipeline-v2.1-lore-metadata-v1",
        "index_version": "idx",
        "entity_vocabulary": ["雷"],
        "items": [
            {"chunk_id": "a", "entities": ["雷"], "dimension_tags": ["physical_rule"], "time_markers": []},
            {"chunk_id": "b", "entities": ["雷"], "dimension_tags": ["physical_rule"], "time_markers": []},
            {"chunk_id": "c", "entities": [], "dimension_tags": ["time"], "time_markers": []},
        ],
    }
    engine = V21Retriever(Dense(), meta)
    structured = {"subject": "雷", "normalized_fact": "亚巴顿寄宿雷的左侧心脏", "dimension": "physical_rule"}

    for method in ("dense", "bm25", "hybrid"):
        report = engine.search_fact(structured, structured["normalized_fact"], method=method, metadata_filter=False, k=2)
        assert report["method"] == method
        assert len(report["results"]) == 2
        assert report["metadata_filter_enabled"] is False
