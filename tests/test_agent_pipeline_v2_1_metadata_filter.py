import numpy as np


def metadata():
    return {
        "schema_version": "agent-pipeline-v2.1-lore-metadata-v1",
        "index_version": "idx",
        "entity_vocabulary": ["亚巴顿", "月城", "雷"],
        "items": [
            {"chunk_id": "c1", "entities": ["亚巴顿", "雷"], "time_markers": [], "dimension_tags": ["physical_rule"]},
            {"chunk_id": "c2", "entities": ["雷"], "time_markers": ["2063年"], "dimension_tags": ["time", "physical_rule"]},
            {"chunk_id": "c3", "entities": ["月城"], "time_markers": ["2064年"], "dimension_tags": ["time"]},
            {"chunk_id": "c4", "entities": [], "time_markers": [], "dimension_tags": ["world_rule"]},
        ],
    }


def fact(text="雷在2063年拥有两颗心脏", subject="雷", dimension="physical_rule"):
    return {"subject": subject, "normalized_fact": text, "dimension": dimension}


def test_progressive_filter_uses_strictest_level_that_meets_floor():
    from agent_pipeline_v2_1.metadata_filter import select_candidates

    result = select_candidates(fact(), metadata(), k=1, min_candidates=1)

    assert result["selected_level"] == "entity+dimension+time"
    assert result["candidate_ids"] == ["c2"]
    assert result["hints"] == {"entities": ["雷"], "dimensions": ["physical_rule"], "times": ["2063年"]}
    assert result["trace"][0] == {"level": "entity+dimension+time", "candidate_count": 1}


def test_progressive_filter_relaxes_and_eventually_falls_back_to_all():
    from agent_pipeline_v2_1.metadata_filter import select_candidates

    relaxed = select_candidates(fact(), metadata(), k=2, min_candidates=2)
    assert relaxed["selected_level"] == "entity+dimension"
    assert relaxed["candidate_ids"] == ["c1", "c2"]
    assert [row["candidate_count"] for row in relaxed["trace"]] == sorted(row["candidate_count"] for row in relaxed["trace"])

    all_rows = select_candidates(fact("陌生人打破未知规则", "陌生人", "causality"), metadata(), k=3, min_candidates=3)
    assert all_rows["selected_level"] == "all"
    assert all_rows["candidate_ids"] == ["c1", "c2", "c3", "c4"]
    assert all_rows["fallback_used"] is True


def test_missing_time_hint_does_not_create_a_fake_time_constraint():
    from agent_pipeline_v2_1.metadata_filter import select_candidates

    result = select_candidates(fact("亚巴顿寄宿在雷的左侧心脏", "亚巴顿"), metadata(), k=1, min_candidates=1)
    assert result["selected_level"] == "entity+dimension"
    assert result["candidate_ids"] == ["c1"]
    assert result["hints"]["times"] == []


def test_filtered_dense_ranking_only_scores_selected_candidates():
    from agent_pipeline_v2_1.metadata_filter import rank_dense_with_metadata

    chunks = [{"id": f"c{i}", "text": str(i), "file": "x", "start_line": i, "end_line": i, "heading_path": []} for i in range(1, 5)]
    vectors = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    query = np.asarray([1.0, 0.0], dtype=np.float32)

    result = rank_dense_with_metadata(vectors, chunks, query, fact(), metadata(), k=2, min_candidates=2)

    assert [row["id"] for row in result["results"]] == ["c1", "c2"]
    assert result["filter"]["selected_level"] == "entity+dimension"
    assert result["scored_candidate_count"] == 2


def test_filter_rejects_metadata_that_does_not_cover_exact_chunk_ids():
    from agent_pipeline_v2_1.metadata_filter import rank_dense_with_metadata

    chunks = [{"id": "wrong", "text": "", "file": "x", "start_line": 1, "end_line": 1, "heading_path": []}]
    vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    try:
        rank_dense_with_metadata(vectors, chunks, query, fact(), metadata(), k=1)
    except ValueError as exc:
        assert "chunk" in str(exc)
    else:
        raise AssertionError("应拒绝metadata与索引不匹配")


def test_oracle_audit_reports_pool_recall_without_changing_the_filter():
    from agent_pipeline_v2_1.metadata_filter import audit_oracle_fixture

    fixture = {"items": [{"fixture_id": "f1", "fact": fact(), "lore": [{"id": "c2"}]}]}
    result = audit_oracle_fixture(fixture, metadata(), k=1, min_candidates=1)

    assert result["oracle_lore_pool_hits"] == 1
    assert result["oracle_lore_pool_recall"] == 1.0
    assert result["rows"][0]["missing_lore_ids"] == []
