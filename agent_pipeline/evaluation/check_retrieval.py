"""Actual CPU/CUDA retrieval observations; no LLM or consistency scoring."""
import hashlib
import json
import uuid
from datetime import datetime,timezone
from pathlib import Path
from agent_pipeline.retrieval import retrieve_events


def main():
    package=Path(__file__).resolve().parents[1];root=package.parent
    directory=package/'reports'/('module_retrieval_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True,exist_ok=False)
    original=package/'reports/retrieval_input_20260918.json'
    input_report=json.loads(original.read_text(encoding='utf-8'))
    (directory/'input.json').write_bytes(original.read_bytes())
    from encoder import Encoder
    from retrieval import Retriever
    from index_store import load_index
    sources=['retrieval.py','__main__.py']
    hashes={s:hashlib.sha256((package/s).read_bytes()).hexdigest() for s in sources}
    observations=[]
    with (directory/'raw.jsonl').open('x',encoding='utf-8') as stream:
        for device in ('cpu','cuda'):
            encoder=Encoder(device=device)
            retriever=Retriever(root/'data/index',encoder)
            result=retrieve_events(input_report,retriever=retriever,device=device)
            reloaded=retrieve_events(input_report,retriever=Retriever(root/'data/index',encoder),device=device)
            def ranks(report):return [(item['event_id'],[(e['id'],e['score']) for e in item['evidence']]) for item in report['items']]
            stable=ranks(result)==ranks(reloaded)
            assert stable
            stream.write(json.dumps(dict(device=device,result=result,reload_identical=stable),ensure_ascii=False)+'\n');stream.flush()
            # Same embedding/index core, focused left/right and unrelated diagnostics.
            probes={q:retriever.search(q) for q in ['雷感觉左胸里的亚巴顿开始躁动。','雷的右侧心脏寄宿拉古艾尔。','番茄炒蛋应该放多少盐？']}
            core=next(c['id'] for c in retriever.metadata['chunks'] if c['file']=='characters.md' and c['heading_path'][-2:]==['雷','生理结构'])
            long_error=None
            try:retriever.search('测试'*600)
            except ValueError as exc:long_error=str(exc)
            assert long_error
            all_results=retrieve_events(input_report,retriever=retriever,k=100)
            assert all(len(item['evidence'])==len(retriever.metadata['chunks']) for item in all_results['items'] if item['status']=='ok')
            observations.append(dict(device=device,status=result['status'],reload_identical=stable,request_seconds=result['request_seconds'],probes=probes,core_example_top5=core in [e['id'] for e in probes['雷感觉左胸里的亚巴顿开始躁动。']],long_query_error=long_error,k100_returns_all=True))
            del retriever,encoder
    report=dict(kind='module_observation_not_formal_benchmark',source_sha256=hashes,cases=observations,snapshot_stable=all(hashlib.sha256((package/s).read_bytes()).hexdigest()==sha for s,sha in hashes.items()),status='completed')
    (directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(directory)


if __name__=='__main__':main()
