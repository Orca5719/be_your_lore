"""Frozen local application baseline; errors and missing facts remain in denominators."""
import argparse
from collections import Counter
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import shutil
import statistics
import tempfile
import time

ROOT=Path(__file__).resolve().parent
LABELS=('一致','矛盾','不确定','缺失/失败')

def score_case(case,preview):
    items=preview.get('items',[])
    evidence={row['id']:row['text'] for row in preview.get('judgment',{}).get('evidence',[])}
    score=dict(claims_total=len(case['claims']),claims_found=0,verdict_correct=0,kind_correct=0,structures_total=len(case.get('structures',[])),structures_found=0,citations_total=0,citations_invalid=0,retrieval_total=0,retrieval_hits=0,citation_core_total=0,citation_core_hits=0,details=[])
    for item in items:
        for citation in item['review'].get('evidence',[]):
            score['citations_total']+=1
            if citation.get('chunk_id') not in evidence or not citation.get('quote') or citation['quote'] not in evidence[citation['chunk_id']]:
                score['citations_invalid']+=1
    for expected in case['claims']:
        matches=[item for item in items if item['proposed']['kind'] not in ('entity','timepoint') and expected['quote'] in item['proposed']['text']]
        found=len(matches)==1
        predicted=matches[0]['review'].get('verdict') if found and matches[0]['review']['status']=='ok' else None
        kind=matches[0]['proposed']['kind'] if found else None
        correct=predicted==expected['verdict']
        kind_correct=kind in expected['kinds']
        score['claims_found']+=found
        score['verdict_correct']+=correct
        score['kind_correct']+=kind_correct
        core=expected.get('evidence_any',[])
        retrieved=[]
        if found:
            for check in preview.get('checks',[]):
                if check['text']==matches[0]['proposed']['text']:
                    retrieved.extend(check['result'].get('evidence',[]))
        retrieval_hit=any(q in row['text'] for q in core for row in retrieved)
        citation_hit=found and any(q in citation.get('quote','') for q in core for citation in matches[0]['review'].get('evidence',[]))
        if core:
            score['retrieval_total']+=1
            score['retrieval_hits']+=retrieval_hit
            score['citation_core_total']+=1
            score['citation_core_hits']+=citation_hit
        score['details'].append(dict(quote=expected['quote'],expected=expected['verdict'],predicted=predicted or '缺失/失败',found=found,kind=kind,kind_correct=kind_correct,correct=correct,retrieval_core_hit=retrieval_hit,citation_core_hit=citation_hit))
    for structure in case.get('structures',[]):
        score['structures_found']+=any(all(item['proposed'].get(k)==v for k,v in structure.items()) for item in items)
    score['complete']=preview.get('status')=='pending_confirmation' and score['verdict_correct']==score['claims_total'] and score['kind_correct']==score['claims_total'] and score['structures_found']==score['structures_total'] and not score['citations_invalid']
    return score

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def report_markdown(report):
    rows=report['cases']; totals=report['totals']
    text=['# 本地世界观工具 baseline v1','', '测试设定，不代表正式 canon。28条短中文输入；26条新组合、2条已见开发回归。标注由实现者运行前编写，没有独立专家复核；新组合仍沿用开发中使用过的资料和语义，不能称严格 held-out。','',f"- 有效预览：{totals['valid_previews']}/{len(rows)}",f"- 核心事实覆盖：{totals['claims_found']}/{totals['claims_total']}",f"- 判定正确（漏项/失败也计入）：{totals['verdict_correct']}/{totals['claims_total']}",f"- 类型符合允许标注：{totals['kind_correct']}/{totals['claims_total']}",f"- 新实体/节点覆盖：{totals['structures_found']}/{totals['structures_total']}",f"- 每例全部事实、类型及必需结构均满足：{totals['complete_cases']}/{len(rows)}",f"- Top-5 核心证据命中：{totals['retrieval_hits']}/{totals['retrieval_total']}（含未提取/失败导致的未检索）",f"- 输出引用核心证据：{totals['citation_core_hits']}/{totals['citation_core_total']}",f"- 引用原文/ID格式有效：{totals['citations_total']-totals['citations_invalid']}/{totals['citations_total']}；有效不保证引用推理正确。",'','## 混淆矩阵','', '| 预期\\输出 | 一致 | 矛盾 | 不确定 | 缺失/失败 |','|---|---:|---:|---:|---:|']
    for label in LABELS[:3]:
        text.append('| '+label+' | '+' | '.join(str(report['confusion'][label][out]) for out in LABELS)+' |')
    text+=['','## 耗时与配置','',f"- 首例：{rows[0]['seconds']:.2f}秒（含Qwen加载，编码器和隔离索引已准备）；Qwen加载{report['qwen_load_seconds']:.2f}秒。不是完整新进程冷启动。",f"- 后续27例单次运行：中位数{report['warm_latency']['median']:.2f}秒，P95 {report['warm_latency']['p95']:.2f}秒；未做重复试验。",f"- CUDA峰值已分配显存最大{max(row['peak_allocated_mib'] for row in rows):.1f} MiB；不包含所有驱动/其他进程显存。",'- BGE CPU FP32，Qwen CUDA NF4双重量化/BF16计算，贪心生成；Top-K=5，每条事实单独检查。','- 不重试、不调提示、不根据结果改标注；原始输出、每项判断、来源、生成token和时间见JSON。',f"- 正式资料及索引指针未变化：{report['original_data_unchanged']}",'','## 分组','', '| 分组 | 全项通过 | 判定正确/事实数 |','|---|---:|---:|']
    for group in ('fresh','development_regression'):
        selected=[r for r in rows if r['group']==group]
        text.append(f"| {group} | {sum(r['score']['complete'] for r in selected)}/{len(selected)} | {sum(r['score']['verdict_correct'] for r in selected)}/{sum(r['score']['claims_total'] for r in selected)} |")
    text+=['','## 逐例','', '| ID | 秒 | 判定正确/事实 | 结构覆盖 | 全项通过 |','|---|---:|---:|---:|---|']
    for row in rows:
        s=row['score'];text.append(f"| {row['id']} | {row['seconds']:.2f} | {s['verdict_correct']}/{s['claims_total']} | {s['structures_found']}/{s['structures_total']} | {'是' if s['complete'] else '否'} |")
    text+=['','## 未通过条目','']
    for row in rows:
        if row['score']['complete']:
            continue
        text.append(f"- **{row['id']}**：{row['text']}")
        if row['preview'].get('error'):
            text.append('  - 处理错误：'+row['preview']['error'])
        for detail in row['score']['details']:
            if not detail['correct'] or not detail['kind_correct']:
                text.append(f"  - {detail['quote']}：预期{detail['expected']}，输出{detail['predicted']}；类型{detail['kind']}，符合标注={detail['kind_correct']}")
        for structure in row.get('structures',[]):
            if not any(all(item['proposed'].get(k)==v for k,v in structure.items()) for item in row['preview'].get('items',[])):
                text.append('  - 缺少结构建议：'+json.dumps(structure,ensure_ascii=False))
    text+=['','边界：证据命中/引用以人工指定原文子串计分，不自动评估原因的语义正确性；并不覆盖指令攻击、长文本、别名、跨语言或复杂时间区间。CPU LLM性能本次未测。']
    return '\n'.join(text)+'\n'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    args=parser.parse_args()
    import torch
    from encoder import Encoder
    from index_store import build_index
    from qwen_judge import QwenJudge,MODEL,REVISION
    import world_entry
    dataset=ROOT/'evaluation/llm_baseline_v1.json'
    data=json.loads(dataset.read_text(encoding='utf-8-sig'))
    for case in data['cases']:
        assert all(c['quote'] in case['text'] for c in case['claims']),case['id']
    files=['characters.md','angels.md','history.md','technology.md']
    watched=[ROOT/'lore'/f for f in files]+[ROOT/'data/index/CURRENT',ROOT/'data/entries.json',ROOT/'data/world_records.json']
    def snapshot():
        return {str(p.relative_to(ROOT)):sha(p) if p.exists() else None for p in watched}
    before=snapshot()
    frozen=[dataset,ROOT/'prompts/world_extraction_v4.txt',ROOT/'prompts/world_review_v2.txt',ROOT/'world_extraction.py',ROOT/'qwen_judge.py',ROOT/'world_entry.py',ROOT/'world_records.py',ROOT/'llm_baseline.py']
    manifest={str(p.relative_to(ROOT)):sha(p) for p in frozen}
    report=dict(schema_version=1,created_at=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),model=MODEL,revision=REVISION,device=args.device,platform=platform.platform(),python=platform.python_version(),packages={name:importlib.metadata.version(name) for name in ['torch','transformers','numpy','bitsandbytes']},manifest=manifest,fixture_hashes={f:sha(ROOT/'lore'/f) for f in files},cases=[],status='running')
    report['hardware']={'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,'gpu_total_mib':torch.cuda.get_device_properties(0).total_memory/2**20 if torch.cuda.is_available() else None}
    output=ROOT/'reports/llm_baseline_v1.json'
    def persist():
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    generations=[]
    generate=QwenJudge._generate
    def measured(judge,*a,**kw):
        try:
            return generate(judge,*a,**kw)
        finally:
            generations.append(dict(judge.last_generation))
    QwenJudge._generate=measured
    with tempfile.TemporaryDirectory(prefix='worldcheck-baseline-') as tmp:
        directory=Path(tmp);lore=directory/'lore';lore.mkdir()
        for f in files:
            shutil.copy2(ROOT/'lore'/f,lore/f)
        started=time.perf_counter();encoder=Encoder(device='cpu');report['encoder_load_seconds']=time.perf_counter()-started
        started=time.perf_counter();index=directory/'index';build_index(lore,index,encoder);report['fixture_index_seconds']=time.perf_counter()-started
        session={'encoder':encoder}
        options=['--lore',str(lore),'--index',str(index),'--store',str(directory/'world.json'),'--legacy-store',str(directory/'legacy.json'),'--device',args.device,'--preview-only','--json']
        for case in data['cases']:
            generations.clear()
            if torch.cuda.is_available() and args.device!='cpu':
                torch.cuda.reset_peak_memory_stats()
            buf=io.StringIO();started=time.perf_counter()
            with contextlib.redirect_stdout(buf):
                code=world_entry.main([case['text']]+options,session)
            seconds=time.perf_counter()-started
            preview=json.loads(buf.getvalue())
            row=dict(**case,code=code,seconds=seconds,preview=preview,score=score_case(case,preview),generations=list(generations),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() and args.device!='cpu' else 0)
            report['cases'].append(row);persist()
            print(json.dumps(dict(id=case['id'],seconds=round(seconds,2),valid=code==0,correct=row['score']['verdict_correct'],total=row['score']['claims_total'],complete=row['score']['complete']),ensure_ascii=False),flush=True)
        assert not (directory/'world.json').exists() and not (directory/'legacy.json').exists()
        report['qwen_load_seconds']=session['judge'].load_seconds
    report['original_data_unchanged']=snapshot()==before
    assert report['original_data_unchanged'],'正式资料在测试期间改变'
    assert all(sha(ROOT/path)==digest for path,digest in manifest.items()),'冻结配置改变，不能合并为一轮baseline'
    rows=report['cases'];totals=Counter()
    for row in rows:
        totals.update({k:v for k,v in row['score'].items() if isinstance(v,int) and k!='complete'})
    totals['valid_previews']=sum(row['code']==0 for row in rows)
    totals['complete_cases']=sum(row['score']['complete'] for row in rows)
    report['totals']=dict(totals)
    confusion={label:{out:0 for out in LABELS} for label in LABELS[:3]}
    for row in rows:
        for detail in row['score']['details']:
            confusion[detail['expected']][detail['predicted']]+=1
    report['confusion']=confusion
    import numpy as np
    warm=[r['seconds'] for r in rows[1:]]
    report['warm_latency']=dict(median=statistics.median(warm),p95=float(np.percentile(warm,95)),min=min(warm),max=max(warm))
    all_generations=[g for row in rows for g in row['generations'] if g.get('seconds',0)>0]
    report['generation_throughput']={'generated_tokens':sum(g['generated_tokens'] for g in all_generations),'seconds':sum(g['seconds'] for g in all_generations)}
    report['generation_throughput']['tokens_per_second']=report['generation_throughput']['generated_tokens']/report['generation_throughput']['seconds']
    report['status']='complete';persist()
    (ROOT/'reports/llm_baseline_v1.md').write_text(report_markdown(report),encoding='utf-8')
    print(json.dumps(report['totals'],ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
