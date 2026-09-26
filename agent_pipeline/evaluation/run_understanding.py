import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime,timezone
from agent_pipeline.understanding import understand


def main():
    parser=argparse.ArgumentParser(description='语义理解模块观察测试；不自动语义评分')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.limit is not None and args.limit<1:parser.error('limit必须为正整数')
    package=Path(__file__).resolve().parents[1]
    dataset=package/'evaluation/understanding_cases_v1.json'
    cases=json.loads(dataset.read_text(encoding='utf-8'))['cases']
    if args.limit:cases=cases[:args.limit]
    directory=package/'reports'/('module_understanding_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'dataset.json').write_bytes(dataset.read_bytes())
    sources=[package/'understanding.py',package/'llm.py',package/'prompts/understanding_v8.txt']
    report=dict(kind='module_observation_not_frozen_benchmark',dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),planned_cases=len(cases),
                source_sha256={p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},cases=[])
    from qwen_judge import QwenJudge,MODEL,REVISION
    llm=QwenJudge(args.device)
    report.update(model=MODEL,revision=REVISION,device=llm.device,model_load_seconds=llm.load_seconds)
    with (directory/'raw.jsonl').open('x',encoding='utf-8') as stream:
        for case in cases:
            start=time.perf_counter()
            try:result=understand(case['text'],llm=llm)
            except (ValueError,RuntimeError,OSError) as exc:result=dict(status='error',error=str(exc),events=[])
            row=dict(id=case['id'],checks=case['checks'],result=result,seconds=time.perf_counter()-start)
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
            report['cases'].append(dict(id=case['id'],status=result['status'],events=len(result['events']),seconds=row['seconds']))
            (directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print(case['id'],result['status'],len(result['events']),'events',f'{row["seconds"]:.2f}s',file=sys.stderr,flush=True)
    report['snapshot_stable']=all(hashlib.sha256((package/name).read_bytes()).hexdigest()==sha for name,sha in report['source_sha256'].items())
    report['status']='completed'
    (directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(directory,flush=True)
    return 0 if report['snapshot_stable'] else 2


if __name__=='__main__':raise SystemExit(main())
