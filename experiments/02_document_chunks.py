"""实验 02：离线读取示例设定，展示编码输入与可追溯来源。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from documents import load_chunks

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lore',type=Path,default=ROOT/'lore')
parser.add_argument('--max-tokens',type=int,default=512)
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-zh-v1.5',cache_dir=ROOT/'.cache/huggingface',local_files_only=True)
chunks = load_chunks(args.lore,tokenizer,args.max_tokens)
for chunk in chunks:
    print(f'\n[{chunk.id}] {chunk.file}:{chunk.start_line}-{chunk.end_line} | {chunk.token_count} tokens')
    print('标题:', ' > '.join(chunk.heading_path))
    print('原文:',chunk.text)
    print('编码输入:',repr(chunk.embedding_text))
print(f'\n共 {len(chunks)} 个片段；预算 {args.max_tokens} tokens（含标题与特殊 token）')
report = ROOT/'reports/chunks.json'
report.parent.mkdir(exist_ok=True)
report.write_text(json.dumps([asdict(c) for c in chunks],ensure_ascii=False,indent=2),encoding='utf-8')
