import pytest

from agent_pipeline_v4.cli import ensure_manifest, check_frozen_files


def test_resume_rejects_changed_input_digest(tmp_path):
    path = tmp_path / "manifest.json"
    ensure_manifest(path, {"schema_version": "test", "identity": {"prompt": "a"}})
    ensure_manifest(path, {"schema_version": "test", "identity": {"prompt": "a"}})
    with pytest.raises(ValueError, match="mismatch"):
        ensure_manifest(path, {"schema_version": "test", "identity": {"prompt": "b"}})


def test_frozen_source_check_normalizes_windows_line_endings(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_bytes(b"first\r\nsecond\r\n")
    import hashlib
    expected = {path: hashlib.sha256(b"first\nsecond\n").hexdigest()}
    check_frozen_files(expected)
    path.write_bytes(b"changed\r\n")
    with pytest.raises(ValueError, match="frozen Benchmark 3"):
        check_frozen_files(expected)
