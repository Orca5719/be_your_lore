"""Real continuous preview regression: qualifier, support, novelty; no saves."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
queries=['雷的左胸会周期性疼痛。','雷的左心脏寄宿亚巴顿。','雷能控制时间。']
original=(ROOT/'data/index/CURRENT').read_bytes()
environment=dict(os.environ,PYTHONUTF8='1')
started=time.perf_counter()
result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'entry.py'),'--interactive','--device','cuda','--preview-only','--json'],input='\n'.join(queries+['/quit'])+'\n',text=True,encoding='utf-8',capture_output=True,env=environment,timeout=240)
rows=[json.loads(line) for line in result.stdout.splitlines() if line.strip()]
expected=['不确定','一致','不确定']
passed=result.returncode==0 and len(rows)==3 and all(row['judgment']['status']=='ok' and row['judgment']['overall_verdict']==verdict for row,verdict in zip(rows,expected))
if len(rows)==3:
    passed=passed and [row['session']['models_reused'] for row in rows]==[False,True,True]
    passed=passed and all(row['session']['qwen_load_seconds']==0 for row in rows[1:])
report=dict(passed=passed,queries=queries,expected=expected,rows=rows,total_wall_seconds=time.perf_counter()-started,stderr=result.stderr,returncode=result.returncode,original_index_unchanged=(ROOT/'data/index/CURRENT').read_bytes()==original,note='三个开发回归样例；单次顺序测量，不是正式 baseline/benchmark，未保存设定')
passed=passed and report['original_index_unchanged']
report['passed']=passed
(ROOT/'reports/continuous_entry_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
for row in rows:
    print(json.dumps(dict(text=row['proposed']['text'],verdict=row['judgment']['overall_verdict'],error=row['judgment'].get('error'),timing=row['judgment'].get('timing'),session=row['session']),ensure_ascii=False),flush=True)
assert passed,'查看 reports/continuous_entry_checks.json'
print('真实连续预览通过，模型加载一次，原索引未改变。')
