from agent_pipeline_v3.paired import run_paired_matrix
from agent_pipeline_v3.report import build_summary, render_markdown


def test_no_model_fixture_covers_paired_and_report_tracks():
    retrieve = lambda method: lambda extraction, top_k: {"method": method, "items": []}
    judge = lambda version: lambda retrieval, batch_size: {"version": version, "items": []}
    matrix = run_paired_matrix([{"id": "E1", "event": "雷触碰墙壁"}], retrieve("dense"), retrieve("hybrid"), judge("v1"), judge("v2.1"))
    assert matrix["status"] == "ok"
    score = {
        "extraction": {"recall": 1.0}, "retrieval": {"recall_at_5": 1.0},
        "judge": {"accuracy": 1.0, "confusion": {}}, "unsupported_reasoning": {"rate": 0.0},
        "end_to_end_conflict": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
    }
    summary = build_summary(score, score, performance={}, attribution={}, manifest={"identity": {"paired_event_digest": matrix["event_digest"]}})
    assert "Agent Pipeline Benchmark 3" in render_markdown(summary)
