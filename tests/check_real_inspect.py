"""真实模型检查：inspect 显示的向量必须与正式编码一致。"""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from encoder import Encoder,QUERY_PROMPT
encoder=Encoder()
text='雷的左心脏寄宿着亚巴顿。'
reports={}
for mode in ['passage','query']:
    report=encoder.inspect_text(text,mode)
    expected=encoder.encode_queries([text])[0] if mode=='query' else encoder.encode_passages([text])[0]
    np.testing.assert_allclose(report['normalized_head'],expected[:8],atol=1e-6)
    assert np.isclose(report['normalized_norm'],1,atol=1e-5)
    assert report['hidden_shape']==[1,len(report['input_ids']),512]
    assert report['cls_shape']==[1,512]
    assert len(report['tokens'])==len(report['attention_mask'])
    assert report['encoding_text']==(QUERY_PROMPT+text if mode=='query' else text)
    reports[mode]=report
try:
    encoder.inspect_text('雷'*500,'query')
except ValueError:
    pass
else:
    raise AssertionError('添加检索提示后的超长输入未被拒绝')
result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'worldcheck.py'),'inspect',text,'--mode','query','--json'],capture_output=True,text=True,encoding='utf-8',timeout=180)
assert result.returncode==0,result.stderr
cli=json.loads(result.stdout)
assert cli['encoding_text']==reports['query']['encoding_text']
np.testing.assert_allclose(cli['normalized_head'],reports['query']['normalized_head'],atol=1e-6)
(ROOT/'reports/inspect_checks.json').write_text(json.dumps({'checks':'passage/query match production vectors, shapes, prompt, length rejection, CLI JSON: passed','reports':reports},ensure_ascii=False,indent=2),encoding='utf-8')
print('inspect 检查通过：正文/查询与正式向量一致，形状、提示、长度拒绝与 CLI JSON 正常。')
print('正文 token 数:',reports['passage']['input_shape'][1],'查询 token 数:',reports['query']['input_shape'][1])
print('正文原向量长度:',reports['passage']['raw_norm'],'归一化长度:',reports['passage']['normalized_norm'])
