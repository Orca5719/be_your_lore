from agent_pipeline_v4.cli_continuous import build_parser, make_manifest, select_wall_seconds


def test_continuous_cli_accepts_run_and_summary_commands():
    parser = build_parser()
    run = parser.parse_args(["run-continuous", "--device", "cuda", "--capacities", "1", "2", "4", "8"])
    assert run.capacities == [1, 2, 4, 8]
    summary = parser.parse_args(["summary-continuous", "--result-dir", "C:/example"])
    assert summary.result_dir.name == "example"


def test_manifest_locks_static_source_and_continuous_implementation(tmp_path):
    old = tmp_path / "static.json"
    engine = tmp_path / "continuous.py"
    old.write_text("old", encoding="utf-8")
    engine.write_text("engine", encoding="utf-8")
    first = make_manifest({"dataset": "a"}, [old, engine])
    engine.write_text("changed", encoding="utf-8")
    second = make_manifest({"dataset": "a"}, [old, engine])
    assert first != second


def test_completed_resume_preserves_wall_but_partial_resume_drops_comparison():
    assert select_wall_seconds(None, existing_count=24, previous_wall=123.0) == 123.0
    assert select_wall_seconds(7.0, existing_count=12, previous_wall=123.0) is None
    assert select_wall_seconds(9.0, existing_count=0, previous_wall=None) == 9.0
