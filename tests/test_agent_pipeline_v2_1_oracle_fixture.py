import json
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "story_benchmark_24_v2.json"
INDEX_METADATA = ROOT / "data" / "index" / "versions" / "0de59066f81246b3b8c7445eddbe877e" / "metadata.json"
CURATION = ROOT / "evaluation" / "agent_pipeline_v2_1_uncertain_oracle_lore.json"


def load_inputs():
    return (
        json.loads(DATASET.read_text(encoding="utf-8")),
        json.loads(INDEX_METADATA.read_text(encoding="utf-8")),
        json.loads(CURATION.read_text(encoding="utf-8")),
    )


def test_oracle_fixture_contains_all_72_balanced_gold_facts():
    from agent_pipeline_v2_1.oracle_fixture import build_oracle_fixture

    dataset, metadata, curation = load_inputs()
    fixture = build_oracle_fixture(dataset, metadata, curation)

    assert fixture["schema_version"] == "agent-pipeline-v2.1-oracle-fixture-v1"
    assert fixture["status"] == "reviewed"
    assert len(fixture["items"]) == 72
    assert Counter(item["expected_verdict"] for item in fixture["items"]) == {
        "consistent": 24,
        "contradiction": 30,
        "uncertain": 18,
    }
    assert len({item["fixture_id"] for item in fixture["items"]}) == 72


def test_oracle_items_use_real_index_chunks_and_never_embed_answers():
    from agent_pipeline_v2_1.oracle_fixture import build_oracle_fixture

    dataset, metadata, curation = load_inputs()
    fixture = build_oracle_fixture(dataset, metadata, curation)
    chunks = {chunk["id"]: chunk for chunk in metadata["chunks"]}

    for item in fixture["items"]:
        assert item["lore"]
        assert "answer" not in item
        assert "predicted_verdict" not in item
        for lore in item["lore"]:
            assert lore == chunks[lore["id"]]


def test_decidable_items_include_a_complete_minimum_evidence_set():
    from agent_pipeline_v2_1.oracle_fixture import build_oracle_fixture

    dataset, metadata, curation = load_inputs()
    fixture = build_oracle_fixture(dataset, metadata, curation)
    gold = {(case["id"], fact["id"]): fact for case in dataset["cases"] for fact in case["gold_facts"]}

    for item in fixture["items"]:
        if item["expected_verdict"] == "uncertain":
            continue
        fact = gold[(item["case_id"], item["gold_fact_id"])]
        if item["oracle_selection"]["source"] == "manual_label_override":
            continue
        actual = {lore["id"] for lore in item["lore"]}
        assert any(set(group) <= actual for group in fact["minimum_evidence_sets"])


def test_uncertain_items_have_manual_nonempty_rationale_and_review_flags_are_preserved():
    from agent_pipeline_v2_1.oracle_fixture import build_oracle_fixture

    dataset, metadata, curation = load_inputs()
    fixture = build_oracle_fixture(dataset, metadata, curation)
    uncertain = [item for item in fixture["items"] if item["expected_verdict"] == "uncertain"]

    assert all(item["oracle_selection"]["source"] == "manual_curation" for item in uncertain)
    assert all(item["oracle_selection"]["rationale"].strip() for item in uncertain)
    assert fixture["review_summary"]["flagged_count"] == 0
    assert fixture["review_summary"]["flags"] == []


def test_fixture_validation_rejects_empty_uncertain_lore():
    from agent_pipeline_v2_1.oracle_fixture import build_oracle_fixture

    dataset, metadata, curation = load_inputs()
    broken = json.loads(json.dumps(curation, ensure_ascii=False))
    first_key = next(iter(broken["facts"]))
    broken["facts"][first_key]["lore_ids"] = []

    with pytest.raises(ValueError, match="uncertain.*lore"):
        build_oracle_fixture(dataset, metadata, broken)


def test_cli_builds_oracle_fixture_without_loading_models(tmp_path):
    from agent_pipeline_v2_1.cli import main

    output = tmp_path / "oracle.json"
    assert main(["build-oracle-fixture", "--output", str(output)]) == 0
    fixture = json.loads(output.read_text(encoding="utf-8"))
    assert len(fixture["items"]) == 72
    assert fixture["review_summary"]["flagged_count"] == 0
