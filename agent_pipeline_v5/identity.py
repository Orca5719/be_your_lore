"""Streaming identity helpers, independent of model packages."""
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def build_identity(config: dict, paths: list[Path]) -> dict:
    return json.loads(json.dumps({'config': config, 'files': {
        str(p.resolve()): sha(p) for p in sorted(set(paths))}}, ensure_ascii=False))
