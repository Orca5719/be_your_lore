"""Check current source files against the indexed source snapshot."""
import hashlib
from pathlib import Path


def validate_sources(metadata):
    root=metadata.get('lore_directory');expected=metadata.get('sources')
    if not isinstance(root,str) or not isinstance(expected,dict):
        raise ValueError('索引缺少资料目录或内容摘要，请重建')
    root=Path(root)
    if not root.is_dir():raise ValueError('索引的资料目录不存在，请重建')
    current={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.suffix.lower() in {'.md','.txt'}}
    if current!=expected:
        added=sorted(current.keys()-expected.keys());removed=sorted(expected.keys()-current.keys())
        changed=sorted(k for k in current.keys()&expected.keys() if current[k]!=expected[k])
        raise ValueError(f'资料已变化，索引过期，请重建；新增={added} 删除={removed} 修改={changed}')
