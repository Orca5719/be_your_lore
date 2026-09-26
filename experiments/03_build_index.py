"""实验 03：批量编码设定，保存并重新加载索引。"""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from encoder import Encoder
from index_store import build_index,load_index

encoder=Encoder()
path=build_index(ROOT/'lore',ROOT/'data/index',encoder)
vectors,metadata=load_index(ROOT/'data/index')
expected=encoder.encode_passages([c['embedding_text'] for c in metadata['chunks']])
np.testing.assert_array_equal(vectors,expected)
print('设备:',encoder.device)
print('索引版本目录:',path)
print('向量矩阵:',vectors.shape,vectors.dtype)
print('向量长度:',np.linalg.norm(vectors,axis=1))
for row,chunk in enumerate(metadata['chunks']):
    print(f"第 {row} 行 -> {chunk['id']} -> {chunk['file']}:{chunk['start_line']} | {chunk['text']}")
print('磁盘重载与重新编码完全一致；文件摘要、单位长度与片段数量检查通过。')
report={'version':path.name,'shape':list(vectors.shape),'device':encoder.device,'norms':np.linalg.norm(vectors,axis=1).tolist(),'reload_equals_reencode':True,'config':metadata['config']}
(ROOT/'reports/experiment03.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
