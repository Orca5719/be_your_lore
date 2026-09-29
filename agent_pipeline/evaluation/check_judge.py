"""Actual isolated judge observations; fixtures are not approved benchmark gold."""
import hashlib
import json
import uuid
from pathlib import Path
from datetime import datetime,timezone
from agent_pipeline import retrieve_events,judge_events


def main():
    package=Path(__file__).resolve().parents[1];root=package.parent
    directory=package/'reports'/('module_judge_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True,exist_ok=False)
    cases=[
        ('J01','亚巴顿','亚巴顿寄宿在雷的左侧心脏里。','observed','consistent'),
        ('J02','亚巴顿','亚巴顿寄宿在雷的右侧心脏里。','observed','contradiction'),
        ('J03','雷','雷拥有两颗心脏。','observed','consistent'),
        ('J04','雷','雷只有一颗心脏。','observed','contradiction'),
        ('J05','雷','雷能够控制时间。','observed','uncertain'),
        ('J06','茱莉亚','茱莉亚入职星球日报。','observed','uncertain'),
        ('J07','雷','雷梦到亚巴顿寄宿在自己的右侧心脏。','dream','uncertain'),
        ('J08','雷','雷认为亚巴顿寄宿在自己的右侧心脏。','belief','uncertain'),
    ]
    (directory/'dataset.json').write_text(json.dumps(dict(kind='assistant_authored_observations_not_gold',cases=cases),ensure_ascii=False,indent=2),encoding='utf-8')
    sources=['judge.py','__main__.py','llm.py','understanding.py','filtering.py','retrieval.py','prompts/judge_v6.txt','prompts/understanding_v8.txt','prompts/filtering_v7.txt','evaluation/check_judge.py','judge.schema.json','event.schema.json']
    hashes={s:hashlib.sha256((package/s).read_bytes()).hexdigest() for s in sources}
    from encoder import Encoder
    from retrieval import Retriever
    from qwen_judge import QwenJudge,MODEL,REVISION
    retriever=Retriever(root/'data/index',Encoder('cpu'))
    llm=QwenJudge('cuda');observations=[]
    with (directory/'raw.jsonl').open('x',encoding='utf-8') as stream:
        for cid,actor,text,modality,expected in cases:
            event=dict(id='E1',actors=[actor],event=text,mental_state=None,explicit=True,modality=modality,conditions=[],source_ids=['S1'],context_ids=[],sources=[dict(id='S1',text=text,start=0,end=len(text))])
            filtered=dict(stage='filtering',status='ok',events=[event],selected_events=[event],ignored_events=[],pending_events=[],decisions=[dict(event_id='E1',decision='keep',reason='助手编写的隔离观察事件')])
            result=judge_events(retrieve_events(filtered,retriever=retriever),llm=llm)
            stream.write(json.dumps(dict(id=cid,expected_draft=expected,result=result),ensure_ascii=False)+'\n');stream.flush()
            observations.append(dict(id=cid,status=result['status'],items=result['items']))
            print(cid,result['status'],result['items'][0].get('verdict'),flush=True)
        # A real previous extraction/filtering/retrieval chain, not a hand-edited fact.
        actual=json.loads((package/'reports/retrieval_story_20260918.json').read_text(encoding='utf-8'))
        result=judge_events(actual,llm=llm)
        stream.write(json.dumps(dict(id='J09_real_chain',result=result),ensure_ascii=False)+'\n');stream.flush()
        observations.append(dict(id='J09_real_chain',status=result['status'],items=result['items']))
        # Preserve complete actual extraction/retrieval, then reuse it to isolate judge.
        for cid,name in [('J10_mooncity','mooncity_judge_fix_v2_20260918.json'),('J11_right_host','right_host_judge_fix_20260918.json')]:
            path=package/'reports'/name
            if not path.exists():continue
            actual=json.loads(path.read_text(encoding='utf-8'))['retrieval']
            result=judge_events(actual,llm=llm)
            stream.write(json.dumps(dict(id=cid,reused_actual_upstream=True,input_file=name,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),result=result),ensure_ascii=False)+'\n');stream.flush()
            observations.append(dict(id=cid,status=result['status'],items=result['items']))
            print(cid,result['status'],[i.get('verdict') for i in result['items']],flush=True)
    report=dict(kind='module_observation_not_formal_benchmark',model=MODEL,revision=REVISION,device=llm.device,model_load_seconds=llm.load_seconds,source_sha256=hashes,cases=observations,status='completed',snapshot_stable=all(hashlib.sha256((package/s).read_bytes()).hexdigest()==sha for s,sha in hashes.items()))
    (directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(directory)


if __name__=='__main__':main()
