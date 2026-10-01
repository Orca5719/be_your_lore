from agent_pipeline_v3_2.cli import _candidate_review


def test_candidate_review_strips_system_namespace():
    review = {
        "event_mapping": {"candidate:SL-001": {"F1": ["E1"]}, "baseline:SL-001": {}},
        "finding_mapping": {},
        "system_event_labels": {},
        "system_finding_labels": {},
        "reasoning_support_labels": {},
    }
    result = _candidate_review(review)
    assert result["event_mapping"] == {"SL-001": {"F1": ["E1"]}}


def test_cli_exposes_lean_extractor_command():
    from agent_pipeline_v3_2.cli import build_parser
    args = build_parser().parse_args(["run-extractor", "--device", "cuda"])
    assert args.command == "run-extractor"
    assert args.device == "cuda"
