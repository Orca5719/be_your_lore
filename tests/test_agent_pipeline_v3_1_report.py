import csv
import json

from agent_pipeline_v3_1.report import render_markdown, write_generation_outputs


def fixture():
    component = {
        "calls": 1, "actual_output_tokens": 100, "field_estimated_tokens": {"events": 30},
        "transport_residual_tokens": 70, "projected_output_tokens": 40,
        "unprojectable_output_tokens": 0, "reduction_tokens": 60, "reduction_ratio": 0.6,
        "actual_decode_time": 10.0, "projected_decode_time": 4.0,
        "theoretical_decode_seconds_saved": 6.0, "projection_kind": "counterfactual_estimate",
        "parse_failures": 0, "verdicts": {},
    }
    return {
        "schema_version": "agent-pipeline-v3.1-generation-audit-v1",
        "source": "x", "stories": 24, "facts": 80,
        "components": {"extractor": dict(component), "judge": dict(component)},
        "retry_audit": {"retry_calls": 31, "totals": {"output_tokens": 3414, "total_time": 289.8}},
        "observations": ["measured"],
    }


def test_generation_report_labels_projection_as_estimate():
    text = render_markdown(fixture())
    assert "Counterfactual estimate" in text
    assert "实测" in text
    assert "不代表真实优化结果" in text


def test_write_generation_outputs_creates_all_part_two_files(tmp_path):
    report = fixture()
    calls = [{"call_id": "L1", "component": "extractor", "output_tokens": 100}]
    fields = [{"component": "extractor", "field": "events", "estimated_tokens": 30}]
    projections = [{"component": "extractor", "actual_output_tokens": 100, "projected_output_tokens": 40}]
    write_generation_outputs(tmp_path, report, calls, fields, projections)
    expected = {
        "generation_audit.json", "generation_audit.md", "calls_audit.csv",
        "retry_taxonomy.csv", "field_token_breakdown.csv", "counterfactual_projection.csv",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    assert json.loads((tmp_path / "generation_audit.json").read_text(encoding="utf-8"))["facts"] == 80
    with (tmp_path / "counterfactual_projection.csv").open(encoding="utf-8-sig") as handle:
        assert list(csv.DictReader(handle))[0]["component"] == "extractor"
