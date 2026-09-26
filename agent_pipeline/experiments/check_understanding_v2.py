import hashlib
import json
import uuid
from datetime import datetime,timezone
from pathlib import Path
from qwen_judge import QwenJudge,MODEL,REVISION
from agent_pipeline.understanding import understand


def main():
    package=Path(__file__).resolve().parents[1]
    directory=package/'reports';directory.mkdir(exist_ok=True)
    path=directory/('understanding_v2_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]+'.json')
    llm=QwenJudge('cuda')
    report=dict(kind='small_learning_experiment_not_benchmark',model=MODEL,revision=REVISION,
                source_sha256={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [package/'understanding.py',package/'llm.py',package/'prompts/understanding_v2.txt']},cases=[])
    examples=[('mental_explicit','雷喝了口水，看到德尔塔的名字时，他感到模糊的熟悉。'),
              ('belief_and_plan','苏晴觉得那人很危险，但她并不认识他。她打算明天离开。'),
              ('literary_transformation','月城的银色手臂如液体一般化开，渗透进实验室的主机中。')]
    for name,text in examples:
        result=understand(text,llm=llm)
        report['cases'].append(dict(id=name,result=result))
        with path.open('w',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
        print(name,result['status'],len(result['events']),'events',flush=True)
    print(path,flush=True)


if __name__=='__main__':main()
