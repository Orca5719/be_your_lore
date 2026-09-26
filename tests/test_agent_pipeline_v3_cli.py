from agent_pipeline_v3.cli import build_parser


def test_cli_commands_and_defaults(tmp_path):
    parser = build_parser()
    assert parser.parse_args(["validate"]).command == "validate"
    end = parser.parse_args(["run-end-to-end"])
    assert (end.device, end.top_k, end.judge_batch_size, end.repeats, end.warmup) == ("cuda", 5, 8, 3, 1)
    for command in ("run-paired", "score", "summary"):
        assert parser.parse_args([command, "--result-dir", str(tmp_path)]).command == command
    assert parser.parse_args(["run-all", "--output", str(tmp_path)]).command == "run-all"
