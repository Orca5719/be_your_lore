"""Module observations, not formal benchmark scoring."""
import hashlib
import json
import sys
import uuid
from datetime import datetime,timezone
from pathlib import Path
from agent_pipeline import filter_events


def main():
    package=Path(__file__).resolve().parents[1]
    dataset=package/'evaluation/filtering_cases_v1.json'
    cases=json.loads(dataset.read_text(encoding='utf-8'))['cases']
    directory=package/'reports'/('module_filtering_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'dataset.json').write_bytes(dataset.read_bytes())
    sources=[package/'filtering.py',package/'llm.py',package/'prompts/filtering_v7.txt']
    hashes={p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    from qwen_judge import QwenJudge,MODEL,REVISION
    llm=QwenJudge('cuda')
    report=dict(kind='module_observation_not_frozen_benchmark',model=MODEL,revision=REVISION,device=llm.device,model_load_seconds=llm.load_seconds,source_sha256=hashes,dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),cases=[])
    with (directory/'raw.jsonl').open('x',encoding='utf-8') as stream:
        for case in cases:
            try:result=filter_events(case['understanding'],llm=llm)
            except (ValueError,RuntimeError,OSError) as exc:result=dict(status='error',error=str(exc),decisions=[])
            stream.write(json.dumps(dict(id=case['id'],result=result),ensure_ascii=False)+'\n');stream.flush()
            report['cases'].append(dict(id=case['id'],status=result['status'],decisions=result['decisions']))
            (directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print(case['id'],result['status'],file=sys.stderr,flush=True)
    report.update(status='completed',snapshot_stable=all(hashlib.sha256((package/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items()))
    (directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(directory)
    return 0 if report['snapshot_stable'] else 2


if __name__=='__main__':raise SystemExit(main())
