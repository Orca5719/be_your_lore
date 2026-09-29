"""实验 04：新剧情 -> 查询向量 -> 点积 -> Top-K 旧设定。"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from encoder import Encoder,QUERY_PROMPT
from retrieval import Retriever,NOTICE

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('query',nargs='?',default='雷感觉左胸里的亚巴顿开始躁动。')
parser.add_argument('--k',type=int,default=5)
parser.add_argument('--json',action='store_true')
args=parser.parse_args()
try:
    if not args.query.strip() or args.k<1:
        raise ValueError('查询不能为空，k 必须为正整数')
    encoder=Encoder()
    retriever=Retriever(ROOT/'data/index',encoder)
    results=retriever.search(args.query,args.k)
except (ValueError,OSError) as error:
    parser.exit(2,f'错误：{error}\n')
report={'query':args.query,'query_input':QUERY_PROMPT+args.query,'index_shape':list(retriever.vectors.shape),'query_shape':[retriever.vectors.shape[1]],'k':args.k,'notice':NOTICE,'results':results}
(ROOT/'reports/experiment04.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
if args.json:
    print(json.dumps(report,ensure_ascii=False,indent=2))
else:
    print('查询:',args.query)
    print('完整查询编码输入:',report['query_input'])
    print(f"[{len(retriever.vectors)}, {retriever.vectors.shape[1]}] @ [{retriever.vectors.shape[1]}] -> [{len(retriever.vectors)}] 个分数")
    for rank,result in enumerate(results,1):
        print(f"\n[{rank}] score={result['score']:.6f} | {result['file']}:{result['start_line']}-{result['end_line']}")
        print('标题:',' > '.join(result['heading_path']))
        print(result['text'])
    print('\n'+NOTICE)
