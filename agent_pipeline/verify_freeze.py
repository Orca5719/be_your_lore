"""Read-only verification for all frozen benchmark snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def _error_result(directory: Path, message: str, file_count: int = 0) -> dict:
    return {
        "name": directory.name,
        "format_version": None,
        "status": "error",
        "files": file_count,
        "changed": [message],
        "archive_ok": False,
    }


def _verify_archive_freeze(directory: Path, manifest: dict) -> dict:
    expected = manifest.get("files", {})
    archive = directory / "snapshot.zip"
    changed: list[str] = []
    archive_ok = archive.is_file() and _sha256(archive.read_bytes()) == manifest.get("archive_sha256")

    try:
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            if len(names) != len(set(names)):
                changed.append("archive:duplicate_entries")
            invalid = sorted(name for name in names if not _valid_archive_name(name))
            changed.extend(f"archive:unsafe_entry:{name}" for name in invalid)

            expected_names = set(expected)
            actual_names = set(names)
            changed.extend(f"archive:missing:{name}" for name in sorted(expected_names - actual_names))
            changed.extend(f"archive:unexpected:{name}" for name in sorted(actual_names - expected_names))
            for name in sorted(expected_names & actual_names):
                if _sha256(zf.read(name)) != expected[name]:
                    changed.append(f"archive:hash_mismatch:{name}")
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        changed.append(f"archive:unreadable:{type(exc).__name__}")

    return {
        "name": manifest.get("name", directory.name),
        "format_version": 2,
        "status": "ok" if archive_ok and not changed else "error",
        "files": len(expected),
        "changed": changed,
        "archive_ok": archive_ok,
    }


def _verify_legacy_freeze(root: Path, directory: Path, manifest: dict) -> dict:
    relocation_file = directory / "document_relocations.json"
    relocations = json.loads(relocation_file.read_text(encoding="utf-8")) if relocation_file.exists() else {}
    for original, destination in relocations.items():
        if (
            original not in manifest["files"]
            or not original.endswith(".md")
            or not destination.startswith("docs/")
            or not destination.endswith(".md")
            or not (root / destination).resolve().is_relative_to(root.resolve())
        ):
            raise ValueError("Invalid frozen document relocation")

    changed = []
    for name, sha in manifest["files"].items():
        path = root / relocations.get(name, name)
        if not path.is_file() or _sha256(path.read_bytes()) != sha:
            changed.append(name)
    archive = directory / "snapshot.zip"
    archive_ok = archive.is_file() and _sha256(archive.read_bytes()) == manifest.get("archive_sha256")
    return {
        "name": manifest.get("name", directory.name),
        "format_version": 1,
        "status": "ok" if not changed and archive_ok else "error",
        "files": len(manifest["files"]),
        "changed": changed,
        "archive_ok": archive_ok,
    }


def verify_directory(root: Path, directory: Path) -> dict:
    """Verify one freeze directory without extracting its snapshot."""
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest.get("files"), dict):
            return _error_result(directory, "manifest:invalid_files")
        if manifest.get("format_version") == 2:
            return _verify_archive_freeze(directory, manifest)
        return _verify_legacy_freeze(root, directory, manifest)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return _error_result(directory, f"manifest:unreadable:{type(exc).__name__}")


def verify() -> dict:
    root = Path(__file__).resolve().parent.parent
    freezes_root = root / "benchmark_freezes"
    directories = sorted(path.parent for path in freezes_root.glob("*/manifest.json"))
    freezes = [verify_directory(root, directory) for directory in directories]
    changed = [
        f"{directory.name}/{item}"
        for directory, result in zip(directories, freezes)
        for item in result["changed"]
    ]
    archive_ok = bool(freezes) and all(result["archive_ok"] for result in freezes)
    return {
        "status": "ok" if freezes and all(result["status"] == "ok" for result in freezes) else "error",
        "files": sum(result["files"] for result in freezes),
        "changed": changed,
        "archive_ok": archive_ok,
        "freezes": freezes,
    }


if __name__ == "__main__":
    result = verify()
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "ok" else 2)
