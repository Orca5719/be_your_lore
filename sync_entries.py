"""Restore generated lore and index from already saved author entries."""
import argparse
from pathlib import Path
import sys
from entries import EntryStore
ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store',type=Path,default=ROOT/'data/entries.json')
    parser.add_argument('--lore',type=Path,default=ROOT/'lore')
    parser.add_argument('--index',type=Path,default=ROOT/'data/index')
    args = parser.parse_args()
    try:
        if not args.store.exists():
            raise ValueError('没有已保存的作者条目')
        from encoder import Encoder
        from index_store import build_index
        EntryStore(args.store).export(args.lore/'_author_entries.md')
        version = build_index(args.lore,args.index,Encoder(device='cpu'))
        print('已同步保存条目，索引版本：'+version.name)
        return 0
    except (ValueError,OSError) as exc:
        print('同步失败：'+str(exc),file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
