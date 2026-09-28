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


def test_cli_exposes_part_two_commands():
    parser = cli.build_parser()
    help_text = parser.format_help()
    assert "validate" in help_text
    assert "audit-retries" in help_text
    assert "audit" in help_text
    assert "summary" in help_text


def test_generation_audit_command_prints_paths(monkeypatch, capsys, tmp_path):
    output = tmp_path / "generation"
    monkeypatch.setattr(cli, "run_generation_audit", lambda source, target: {"stories": 24, "facts": 80})
    code = cli.main(["audit", "--result-dir", str(tmp_path), "--output", str(output)])
    text = capsys.readouterr().out
    assert code == 0
    assert f"AUDIT_DIR={output.resolve()}" in text
    assert f"SUMMARY={output.resolve() / 'generation_audit.md'}" in text


def test_summary_does_not_load_tokenizer(monkeypatch, capsys, tmp_path):
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    (audit_dir / "generation_audit.json").write_text(json.dumps({
        "stories": 24, "facts": 80, "components": {}, "observations": []
    }), encoding="utf-8")
    assert cli.main(["summary", "--result-dir", str(audit_dir)]) == 0
    assert "SUMMARY=" in capsys.readouterr().out
