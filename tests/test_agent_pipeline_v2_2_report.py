from agent_pipeline_v2_2.report import build_summary, render_markdown


def score(value):
    return {"extraction": {"recall": value}, "retrieval": {"recall_at_5": value}, "judge": {"accuracy": value, "confusion": {}}, "unsupported_reasoning": {"rate": 0.0}, "end_to_end_conflict": {"precision": value, "recall": value, "f1": value}}


def test_report_contains_metrics_deltas_attribution_and_limitations():
    report = build_summary(score(.5), score(.75), performance={"candidate": {"median_total_seconds": 1}}, attribution={"candidate": {"counts": {"judge_fn": 1}}}, manifest={"identity": {"repeats": 3}})
    assert report["deltas"]["end_to_end_f1"]["absolute"] == .25
    text = render_markdown(report)
    for term in ("Extraction Recall", "Retrieval Recall@5", "Judge Accuracy", "End-to-End F1", "Error attribution", "Performance", "Limitations"):
        assert term in text
