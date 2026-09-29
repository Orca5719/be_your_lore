import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_v2_1_package_has_independent_identity_and_defaults():
    import agent_pipeline_v2
    import agent_pipeline_v2_1
    from agent_pipeline_v2_1.contracts import BenchmarkConfig

    config = BenchmarkConfig()

    assert agent_pipeline_v2_1 is not agent_pipeline_v2
    assert agent_pipeline_v2_1.VERSION == "2.1"
    assert config.judge_batch_size == 8
    assert config.repeats == 3
    assert config.report_namespace == "agent_pipeline_v2_1"


def test_retrieval_config_exposes_approved_public_modes():
    from agent_pipeline_v2_1.contracts import RetrievalConfig

    assert RetrievalConfig(method="dense", metadata_filter=False, k=5).method == "dense"
    assert RetrievalConfig(method="bm25", metadata_filter=True, k=5).method == "bm25"
    assert RetrievalConfig(method="hybrid", metadata_filter=True, k=10).method == "hybrid"

    with pytest.raises(ValueError, match="method"):
        RetrievalConfig(method="reranker", metadata_filter=True, k=5)
    with pytest.raises(ValueError, match="k"):
        RetrievalConfig(method="dense", metadata_filter=False, k=0)


def test_v2_1_cli_info_is_separate_and_machine_readable():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "agent_pipeline_v2_1_benchmark.py"), "info", "--json"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {
        "schema_version": "agent-pipeline-v2.1-benchmark-info-v1",
        "package": "agent_pipeline_v2_1",
        "version": "2.1",
        "judge_batch_size": 8,
        "repeats": 3,
            "implemented_stages": ["scaffold", "judge_contract", "oracle_fixture", "oracle_benchmark", "lore_metadata", "metadata_filter", "bm25", "rrf_hybrid", "retrieval_benchmark_2_1B", "story_diagnostic_24"],
        "planned_benchmarks": ["2.1A", "2.1B", "2.1C"],
    }
