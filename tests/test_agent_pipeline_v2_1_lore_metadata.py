import json


def chunks():
    return [
        {
            "id": "c1",
            "text": "雷拥有两颗心脏。左侧心脏寄宿亚巴顿，右侧心脏寄宿拉古艾尔。",
            "file": "characters.md",
            "start_line": 6,
            "end_line": 6,
            "heading_path": ["人物设定", "雷", "生理结构"],
        },
        {
            "id": "c2",
            "text": "2063年大地震摧毁旧城区。雷在废墟中昏迷。",
            "file": "history.md",
            "start_line": 6,
            "end_line": 6,
            "heading_path": ["历史事件", "2063年", "大地震"],
        },
        {
            "id": "c3",
            "text": "公众知道德尔塔的代号，但不知道装甲下的真实姓名。",
            "file": "characters.md",
            "start_line": 21,
            "end_line": 21,
            "heading_path": ["人物设定", "德尔塔", "公众身份"],
        },
        {
            "id": "c4",
            "text": "亚巴顿在大地震时提前降临。",
            "file": "angels.md",
            "start_line": 6,
            "end_line": 6,
            "heading_path": ["天使设定", "亚巴顿", "提前降临"],
        },
        {
            "id": "c5",
            "text": "拉古艾尔可以加速伤口愈合。",
            "file": "angels.md",
            "start_line": 11,
            "end_line": 11,
            "heading_path": ["天使设定", "拉古艾尔", "治疗能力"],
        },
    ]


def test_rule_metadata_is_deterministic_and_keeps_provenance():
    from agent_pipeline_v2_1.lore_metadata import build_lore_metadata

    one = build_lore_metadata(chunks(), index_version="idx")
    two = build_lore_metadata(list(reversed(chunks())), index_version="idx")

    assert one == two
    assert one["schema_version"] == "agent-pipeline-v2.1-lore-metadata-v1"
    assert one["generator"]["uses_llm"] is False
    assert one["index_version"] == "idx"
    assert one["chunk_count"] == 5
    assert one["metadata_digest"]
    assert one["summary"]["coverage"]["dimension_tags"]["count"] == 4
    assert [row["chunk_id"] for row in one["items"]] == ["c1", "c2", "c3", "c4", "c5"]


def test_metadata_uses_heading_vocabulary_and_explicit_text_only():
    from agent_pipeline_v2_1.lore_metadata import build_lore_metadata

    result = build_lore_metadata(chunks(), index_version="idx")
    by_id = {row["chunk_id"]: row for row in result["items"]}

    assert by_id["c1"]["primary_scope"] == "雷"
    assert by_id["c1"]["category"] == "生理结构"
    assert by_id["c1"]["entities"] == ["亚巴顿", "拉古艾尔", "雷"]
    assert "physical_rule" in by_id["c1"]["dimension_tags"]
    assert by_id["c2"]["time_markers"] == ["2063年"]
    assert "雷" in by_id["c2"]["entities"]
    assert by_id["c2"]["primary_scope"] is None
    assert "time" in by_id["c2"]["dimension_tags"]
    assert {"character_knowledge", "identity"} <= set(by_id["c3"]["dimension_tags"])


def test_metadata_does_not_invent_unknown_entities_or_locations():
    from agent_pipeline_v2_1.lore_metadata import build_lore_metadata

    result = build_lore_metadata(chunks(), index_version="idx")
    row = next(row for row in result["items"] if row["chunk_id"] == "c2")

    assert "旧城区" not in row["entities"]
    assert row["locations"] == []
    assert row["provenance"]["entities"] == "heading_vocabulary_exact_match"
    assert row["provenance"]["locations"] == "not_inferred"


def test_generated_snapshot_matches_source_chunks(tmp_path):
    from agent_pipeline_v2_1.lore_metadata import build_lore_metadata, validate_lore_metadata

    value = build_lore_metadata(chunks(), index_version="idx")
    validate_lore_metadata(value, chunks(), expected_index_version="idx")
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    validate_lore_metadata(loaded, chunks(), expected_index_version="idx")
