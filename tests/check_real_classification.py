"""Real classification development cases, model loaded once, no canon writes."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qwen_judge import QwenJudge
ROOT = Path(__file__).resolve().parents[1]
judge = QwenJudge('cuda')
cases = [
    ('雷能控制时间。','雷','能力',None),
    ('雷拥有两颗心脏。','雷','生理结构',None),
    ('第三话，雷失去了长剑。','雷','经历','第三话'),
    ('大地震后，雷失去了记忆。','雷','经历','大地震后'),
    ('他能控制时间。',None,'能力',None),
]
rows=[]
for text,entity,category,time in cases:
    result=judge.classify(text,['雷','莉娅'],['能力','生理结构','经历','身份','关系','装备','规则','timeline'])
    passed=all(result[field]==expected for field,expected in [('entity',entity),('category',category),('event_time',time)])
    rows.append(dict(text=text,result=result,expected=dict(entity=entity,category=category,event_time=time),passed=passed))
    print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
report=dict(cases=rows,passed=all(row['passed'] for row in rows),note='5 个开发案例，不是 baseline；未保存资料')
(ROOT/'reports/classification_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
assert report['passed'], '查看报告中的错误分类'
