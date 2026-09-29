"""Offline report observations on retained real pipeline outputs; not a benchmark."""
import hashlib
import json
import uuid
from datetime import datetime,timezone
from pathlib import Path
from agent_pipeline import build_report


def main():
    package=Path(__file__).resolve().parents[1]
    directory=package/'reports'/('module_report_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True,exist_ok=False)
    sources=['report.py','__main__.py','report.schema.json','evaluation/check_report.py','filtering.py','prompts/filtering_v7.txt','judge.py','prompts/judge_v6.txt']
    hashes={name:hashlib.sha256((package/name).read_bytes()).hexdigest() for name in sources}
    cases=[('R01_full_pipeline','report_full_story_20260918.json'),('R02_right_host','right_host_judge_final_20260918.json'),
           ('R03_mooncity','mooncity_judge_final_20260918.json'),('R04_partial_upstream','mooncity_judge_fix_20260918.json')]
    for cid,name in [('R05_feedback_mooncity','report_mooncity_feedback_final_20260919.json'),('R06_feedback_right','report_right_feedback_final_20260919.json')]:
        if (package/'reports'/name).exists():cases.append((cid,name))
    observations=[]
    with (directory/'raw.jsonl').open('x',encoding='utf-8') as stream:
        for cid,name in cases:
            path=package/'reports'/name
            try:
                raw=path.read_bytes();original=json.loads(raw)
                judged=original['judge'] if original['stage']=='report' else original
                result=build_report(judged)
                record=dict(id=cid,input_file=name,input_sha256=hashlib.sha256(raw).hexdigest(),result=result)
                observations.append(dict(id=cid,status=result['status'],summary=result['summary']))
            except (ValueError,OSError) as exc:
                record=dict(id=cid,input_file=name,error=str(exc));observations.append(dict(id=cid,status='observation_error',error=str(exc)))
            stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
            print(cid,observations[-1],flush=True)
    report=dict(kind='offline_presentation_observations_not_accuracy_benchmark',sources_sha256=hashes,cases=observations,
                snapshot_stable=all(hashlib.sha256((package/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items()))
    (directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(directory)


if __name__=='__main__':main()
