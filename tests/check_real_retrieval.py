"""真实模型检索检查：不加入快速单元测试的默认运行。"""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from encoder import Encoder
from retrieval import Retriever

encoder=Encoder()
retriever=Retriever(ROOT/'data/index',encoder)
query='雷感觉左胸里的亚巴顿开始躁动。'
first=retriever.search(query)
second=Retriever(ROOT/'data/index',encoder).search(query)
assert first==second,'重载后查询结果不一致'
assert any('左侧心脏寄宿亚巴顿' in r['text'] for r in first),'未找回核心设定'
errors={}
for label,text in [('empty','   '),('too_long','雷'*600)]:
    try:
        retriever.search(text)
    except ValueError as error:
        errors[label]=str(error)
    else:
        raise AssertionError(f'{label} 未被拒绝')
unrelated=retriever.search('番茄炒蛋需要放多少盐？')
assert len(unrelated)==min(5,len(retriever.vectors))
report={'core_evidence_found':True,'reload_results_identical':True,'errors':errors,'unrelated_query':'番茄炒蛋需要放多少盐？','unrelated_results':unrelated}
(ROOT/'reports/experiment04_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('核心证据找回、重载结果一致、空查询和超长查询拒绝：通过')
print('无关查询仍返回',len(unrelated),'个片段；分数:',[round(r['score'],6) for r in unrelated])
print('本脚本是边界检查；正式检索评测运行 evaluate.py。')

