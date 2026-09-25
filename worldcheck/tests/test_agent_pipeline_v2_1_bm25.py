def chunks():
    return [
        {
            "id": "left",
            "text": "亚巴顿寄宿在雷的左侧心脏里。",
            "file": "angels.md",
            "start_line": 1,
            "end_line": 1,
            "heading_path": ["天使", "亚巴顿", "左侧宿主"],
        },
        {
            "id": "right",
            "text": "拉古艾尔寄宿在雷的右侧心脏里。",
            "file": "angels.md",
            "start_line": 2,
            "end_line": 2,
            "heading_path": ["天使", "拉古艾尔", "右侧宿主"],
        },
        {
            "id": "history",
            "text": "2063年大地震摧毁了旧城区。",
            "file": "history.md",
            "start_line": 3,
            "end_line": 3,
            "heading_path": ["历史", "2063年", "大地震"],
        },
    ]


def test_chinese_tokenizer_is_deterministic_and_keeps_unigrams_bigrams():
    from agent_pipeline_v2_1.bm25 import tokenize_zh

    assert tokenize_zh("左胸 A-12") == ["左", "胸", "左胸", "a", "12"]
    assert tokenize_zh("左胸 A-12") == tokenize_zh("左胸 A-12")


def test_bm25_ranks_exact_side_and_entity_above_neighboring_lore():
    from agent_pipeline_v2_1.bm25 import BM25Retriever

    retriever = BM25Retriever(chunks())
    rows = retriever.search("亚巴顿位于雷的左侧心脏", k=3)

    assert rows[0]["id"] == "left"
    assert rows[0]["score"] > rows[1]["score"]
    assert "亚巴" in rows[0]["matched_terms"]
    assert rows[0]["rank"] == 1


def test_bm25_can_rank_heading_terms_and_numbers():
    from agent_pipeline_v2_1.bm25 import BM25Retriever

    rows = BM25Retriever(chunks()).search("2063年发生大地震", k=1)
    assert [row["id"] for row in rows] == ["history"]


def test_bm25_candidate_filter_is_optional_and_auditable():
    from agent_pipeline_v2_1.bm25 import BM25Retriever

    retriever = BM25Retriever(chunks())
    result = retriever.search_with_report("亚巴顿位于雷的左侧心脏", k=2, candidate_ids=["right", "history"])

    assert result["method"] == "bm25"
    assert result["candidate_count"] == 2
    assert result["candidate_filter_applied"] is True
    assert [row["id"] for row in result["results"]] == ["right", "history"]


def test_bm25_zero_overlap_is_stable_and_errors_are_clear():
    from agent_pipeline_v2_1.bm25 import BM25Retriever

    retriever = BM25Retriever(chunks())
    rows = retriever.search("xyz", k=5)
    assert [row["id"] for row in rows] == ["left", "right", "history"]
    assert all(row["score"] == 0.0 for row in rows)

    for query in ("", "   "):
        try:
            retriever.search(query)
        except ValueError as exc:
            assert "查询" in str(exc)
        else:
            raise AssertionError("空查询必须失败")

    try:
        retriever.search("雷", candidate_ids=["missing"])
    except ValueError as exc:
        assert "candidate" in str(exc)
    else:
        raise AssertionError("未知candidate id必须失败")

