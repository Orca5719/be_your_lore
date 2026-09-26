"""Same-input reproduction and diverse observations, never automatic accuracy."""
import hashlib
import json
import uuid
import argparse
from datetime import datetime,timezone
from pathlib import Path
from agent_pipeline import understand,filter_events


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true',help='只复测问题句与普通日常两段；不代表完整观察集')
    args=parser.parse_args()
    package=Path(__file__).resolve().parents[1]
    directory=package/'reports'/('reported_failure_fix_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True,exist_ok=False)
    sources=['understanding.py','filtering.py','llm.py','prompts/understanding_v8.txt','prompts/filtering_v7.txt']
    hashes={s:hashlib.sha256((package/s).read_bytes()).hexdigest() for s in sources}
    fixtures=json.loads((package/'evaluation/filtering_cases_v1.json').read_text(encoding='utf-8'))
    stories=[('R01','雷喝下杯里的水，随即毒发倒地。他的左心脏中，亚巴顿开始躁动。'),
             ('R02','雷喝了口普通水，然后坐下来系鞋带。窗帘随风轻轻摆动。'),
             ('R03','月城的银色手臂如液体一般化开，渗透进了实验室的主机中，随后屏幕开始发生变化，各种文件飞速闪过，只留下一张图像——大地震震中')]
    if args.quick:
        stories=stories[:2]
        fixtures=dict(cases=[])
    (directory/'dataset.json').write_text(json.dumps(dict(stories=stories,fixtures=fixtures),ensure_ascii=False,indent=2),encoding='utf-8')
    from qwen_judge import QwenJudge,MODEL,REVISION
    llm=QwenJudge('cuda')
    report=dict(kind='assistant_observation_not_formal_benchmark',model=MODEL,revision=REVISION,device=llm.device,source_sha256=hashes,model_load_seconds=llm.load_seconds,cases=[])
    with (directory/'raw.jsonl').open('x',encoding='utf-8') as stream:
        for cid,text in stories:
            result=filter_events(understand(text,llm=llm),llm=llm)
            stream.write(json.dumps(dict(id=cid,result=result),ensure_ascii=False)+'\n');stream.flush()
            report['cases'].append(dict(id=cid,status=result['status'],decisions=result['decisions']))
            print(cid,result['status'],flush=True)
        for case in fixtures['cases']:
            result=filter_events(case['understanding'],llm=llm)
            stream.write(json.dumps(dict(id=case['id'],result=result),ensure_ascii=False)+'\n');stream.flush()
            report['cases'].append(dict(id=case['id'],status=result['status'],decisions=result['decisions']))
            print(case['id'],result['status'],flush=True)
    report.update(status='completed',snapshot_stable=all(hashlib.sha256((package/s).read_bytes()).hexdigest()==sha for s,sha in hashes.items()))
    (directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(directory)


if __name__=='__main__':main()
