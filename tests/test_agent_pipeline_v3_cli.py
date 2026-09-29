from pathlib import Path

from agent_pipeline_v3.cli import build_parser, validate_inputs


def test_cli_has_fixed_profile_commands_and_cuda_default():
    parser = build_parser()
    validate = parser.parse_args(["validate"])
    run = parser.parse_args(["run"])
    summary = parser.parse_args(["summary", "--result-dir", "x"])
    assert validate.command == "validate"
    assert run.command == "run"
    assert run.device == "cuda"
    assert run.output is None
    assert not hasattr(run, "repeats")
    assert summary.result_dir == Path("x")


def test_validate_inputs_finds_fixed_24_story_quality_baseline():
    result = validate_inputs()
    assert result["status"] == "ok"
    assert result["cases"] == 24
    assert result["gold_facts"] == 72
    assert result["quality_reference_cases"] == 24
