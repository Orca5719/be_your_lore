"""Diagnostic baseline with manually specified claims; no extraction and no persistence."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time
from llm_baseline import sha,LABELS
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda');args=p.parse_args()
    import torch
    from encoder import Encoder
    from index_store import build_index
    from retrieval import Retriever
    from qwen_judge import QwenJudge,MODEL,REVISION
    from world_records import validate_plan
    dataset=ROOT/'evaluation/llm_judge_baseline_v1.json';data=json.loads(dataset.read_text(encoding='utf-8'))
    for case in data['cases']:
        validate_plan({'records':[case['record']]},case['source_input'])
    frozen=[dataset,ROOT/'prompts/world_review_v2.txt',ROOT/'qwen_judge.py',ROOT/'world_records.py',ROOT/'llm_judge_baseline.py']
    manifest={str(f.relative_to(ROOT)):sha(f) for f in frozen}
    files=['characters.md','angels.md','history.md','technology.md']; watched=[ROOT/'lore'/f for f in files]+[ROOT/'data/index/CURRENT',ROOT/'data/world_records.json',ROOT/'data/entries.json']
    before={str(f):sha(f) if f.exists() else None for f in watched}
    report=dict(status='running',model=MODEL,revision=REVISION,manifest=manifest,note=data['note'],cases=[],device=args.device,fixture_hashes={f:sha(ROOT/'lore'/f) for f in files})
    output=ROOT/'reports/llm_judge_baseline_v1.json'
    def persist():output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='worldcheck-judge-baseline-') as tmp:
        d=Path(tmp);lore=d/'lore';lore.mkdir()
        for f in files:shutil.copy2(ROOT/'lore'/f,lore/f)
        encoder=Encoder(device='cpu');build_index(lore,d/'index',encoder);ret=Retriever(d/'index',encoder);judge=QwenJudge(args.device);report['qwen_load_seconds']=judge.load_seconds
        for case in data['cases']:
            record=case['record'];start=time.perf_counter()
            if judge.device=='cuda':torch.cuda.reset_peak_memory_stats()
            query=' '.join(filter(None,[record['entity'],record['target'],record['time'],record['text']]))
            results=ret.search(query)
            answer=judge.check_world(record['text'],[record],results)
            predicted=answer.get('overall_verdict') if answer['status']=='ok' else '缺失/失败'
            core=case['evidence_any']
            citations=[c for f in answer.get('findings',[]) for c in f['evidence']]
            evidence={r['id']:r['text'] for r in results}
            row=dict(**case,predicted=predicted,correct=predicted==case['expected'],result=answer,seconds=time.perf_counter()-start,retrieval_core_hit=any(q in r['text'] for q in core for r in results),citation_core_hit=any(q in c['quote'] for q in core for c in citations),citations_invalid=sum(c['chunk_id'] not in evidence or c['quote'] not in evidence[c['chunk_id']] for c in citations),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if judge.device=='cuda' else 0)
            report['cases'].append(row);persist()
            print(json.dumps(dict(id=case['id'],predicted=predicted,expected=case['expected'],seconds=round(row['seconds'],2)),ensure_ascii=False),flush=True)
    assert all(sha(ROOT/f)==h for f,h in manifest.items())
    report['original_data_unchanged']=before=={str(f):sha(f) if f.exists() else None for f in watched};assert report['original_data_unchanged']
    rows=report['cases'];core_rows=[r for r in rows if r['evidence_any']]
    report['totals']=dict(correct=sum(r['correct'] for r in rows),total=len(rows),valid=sum(r['result']['status']=='ok' for r in rows),retrieval_hits=sum(r['retrieval_core_hit'] for r in core_rows),retrieval_total=len(core_rows),citation_core_hits=sum(r['citation_core_hit'] for r in core_rows),citations_invalid=sum(r['citations_invalid'] for r in rows))
    report['confusion']={label:{out:sum(r['expected']==label and r['predicted']==out for r in rows) for out in LABELS} for label in LABELS[:3]}
    import statistics
    import numpy as np
    report['latency']=dict(median=statistics.median(r['seconds'] for r in rows),p95=float(np.percentile([r['seconds'] for r in rows],95)))
    report['status']='complete';persist()
    t=report['totals'];text=['# 人工拆分后的检索与判断诊断 baseline','',data['note'],'',f"- 判定正确：{t['correct']}/{t['total']}",f"- 输出协议有效：{t['valid']}/{t['total']}",f"- Top5核心证据：{t['retrieval_hits']}/{t['retrieval_total']}",f"- 引用核心证据：{t['citation_core_hits']}/{t['retrieval_total']}",f"- 无效引用：{t['citations_invalid']}",f"- 模型加载另计 {report['qwen_load_seconds']:.2f} 秒；每条检索+判断中位数 {report['latency']['median']:.2f} 秒，P95 {report['latency']['p95']:.2f} 秒，单次测量。",'','| ID | 预期 | 输出 | 原因 |','|---|---|---|---|']
    for r in rows:
        reason='；'.join(f['reason'] for f in r['result'].get('findings',[])) or r['result'].get('error','')
        text.append(f"| {r['id']} | {r['expected']} | {r['predicted']} | {reason.replace('|','/')} |")
    text+=['','原始输出、证据、版本、内容摘要和逐例耗时见对应 JSON。没有自动评估原因语义，不能将协议有效等同于判断可靠。']
    (ROOT/'reports/llm_judge_baseline_v1.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(t,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
