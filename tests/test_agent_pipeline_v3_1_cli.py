import json
from pathlib import Path

from agent_pipeline_v3_1 import cli


def test_validate_command_prints_read_only_validation(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "validate_result", lambda path: {"status": "ok", "stories": 24, "calls": 130})
    code = cli.main(["validate", "--result-dir", str(tmp_path)])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["calls"] == 130


def test_audit_retries_prints_output_and_summary(monkeypatch, capsys, tmp_path):
    output = tmp_path / "out"
    monkeypatch.setattr(cli, "audit_retries", lambda source, target: {"retry_calls": 31, "totals": {"output_tokens": 100}})
    code = cli.main(["audit-retries", "--result-dir", str(tmp_path), "--output", str(output)])
    text = capsys.readouterr().out
    assert code == 0
    assert f"AUDIT_DIR={output.resolve()}" in text
    assert '"retry_calls": 31' in text


def test_cli_has_only_part_one_commands():
    parser = cli.build_parser()
    help_text = parser.format_help()
    assert "validate" in help_text
    assert "audit-retries" in help_text
    assert " counterfactual" not in help_text
