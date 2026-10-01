from agent_pipeline_v3_2.extractor_report import render_extractor_audit


def test_extractor_report_names_costs_and_closure():
    bucket = {"calls": 0, "stories": 0, "windows": 0, "input_tokens": 0, "output_tokens": 0, "prefill_time": 0.0, "decode_time": 0.0, "total_time": 0.0, "time_share": 0.0, "recovered_calls": 0, "case_ids": []}
    report = {
        "stories": 24, "extractor_calls": 103, "totals": bucket,
        "cost_classes": {name: dict(bucket) for name in ("pure_waste", "recovery_cost", "effective_cost")},
        "failure_categories": {}, "outcomes": {}, "recommendations": [],
        "closure": {"calls_closed": True, "tokens_closed": True, "time_closed": True},
    }
    text = render_extractor_audit(report)
    assert "pure_waste" in text
    assert "Calls closed: `True`" in text
    assert "没有加载模型" in text
