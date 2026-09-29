"""Story-level metrics from explicit reviewed semantic matches, never LLM self-grading."""
import statistics
import math

LABELS=('一致','矛盾','不确定')


def ratio(a,b):
    return a/b if b else None


def prf(tp,fp,fn):
    return dict(tp=tp,fp=fp,fn=fn,precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),f1=ratio(2*tp,2*tp+fp+fn))


def unique(rows,key):
    result={}
    for row in rows:
        value=row[key]
        if value in result:raise ValueError('重复编号：'+value)
        result[value]=row
    return result


def score_results(dataset,rows,review,k=5):
    cases=unique(dataset['cases'],'id')
    outputs=unique(rows,'case_id')
    reviewed=unique(review['cases'],'case_id')
    if set(outputs)-set(cases) or set(reviewed)!=set(cases):raise ValueError('运行或审核案例编号不匹配')
    if any(row.get('reviewed') is not True for row in reviewed.values()):
        raise ValueError('语义配对尚未审核完成；不能生成检测指标')
    etp=efp=efn=ctp=cfp=cfn=correct=gold_total=evidence_hits=evidence_total=0
    daily_total=daily_clean=daily_fp=failed_cases=0
    stp=sfp=sfn=0
    confusion={label:{col:0 for col in (*LABELS,'missing_or_error')} for label in LABELS}
    for cid,case in cases.items():
        result=outputs.get(cid,{}).get('result',{})
        extraction=result.get('extraction',{})
        preds=unique(extraction.get('facts',[]),'id')
        golds=unique(case['gold_facts'],'id')
        items=unique([dict(x,id=x['fact']['id']) for x in result.get('items',[])],'id')
        if set(items)-set(preds):raise ValueError('检索/判断条目没有对应提取事实')
        pairs=reviewed[cid].get('matches')
        if not isinstance(pairs,list):raise ValueError('matches必须是列表')
        matched_pred=set();matched_gold=set();mapping={}
        for pair in pairs:
            pid=pair.get('prediction_id');gid=pair.get('gold_id')
            if pid not in preds or gid not in golds or pid in matched_pred or gid in matched_gold:
                raise ValueError('配对编号无效或不满足一对一：'+cid)
            matched_pred.add(pid);matched_gold.add(gid);mapping[gid]=pid
        etp+=len(pairs);efp+=len(preds)-len(pairs);efn+=len(golds)-len(pairs)
        gold_total+=len(golds)
        failed_cases+=result.get('status')!='ok'
        positive_preds={pid for pid,item in items.items() if item.get('judgement',{}).get('status')=='ok' and item['judgement'].get('verdict')=='矛盾'}
        gold_conflicts={gid for gid,gold in golds.items() if gold['expected_verdict']=='矛盾'}
        found_conflicts={mapping[gid] for gid in gold_conflicts if gid in mapping and mapping[gid] in positive_preds}
        ctp+=len(found_conflicts);cfp+=len(positive_preds)-len(found_conflicts);cfn+=len(gold_conflicts)-len(found_conflicts)
        stp+=bool(gold_conflicts and positive_preds)
        sfp+=bool(positive_preds and not gold_conflicts)
        sfn+=bool(gold_conflicts and not positive_preds)
        for gid,gold in golds.items():
            item=items.get(mapping.get(gid),{})
            answer=item.get('judgement',{})
            predicted=answer.get('verdict') if answer.get('status')=='ok' else None
            if predicted not in LABELS:predicted='missing_or_error'
            expected=gold['expected_verdict']
            confusion[expected][predicted]+=1
            correct+=predicted==expected
            alternatives=gold.get('acceptable_evidence_sets',[])
            if alternatives:
                evidence_total+=1
                retrieved={row['id'] for row in item.get('evidence',[])[:k]} if item.get('status')=='ok' else set()
                evidence_hits+=any(set(group)<=retrieved for group in alternatives)
        if not golds:
            daily_total+=1;daily_fp+=len(preds)
            daily_clean+=result.get('status')=='ok' and extraction.get('status')=='ok' and not preds
    return dict(extraction=prf(etp,efp,efn),conflict=prf(ctp,cfp,cfn),story_conflict=prf(stp,sfp,sfn),
        classification=dict(correct=correct,gold_total=gold_total,accuracy=ratio(correct,gold_total),confusion=confusion),
        evidence=dict(hits=evidence_hits,gold_total=evidence_total,k=k,recall=ratio(evidence_hits,evidence_total)),
        daily=dict(cases=daily_total,clean_empty=daily_clean,clean_empty_rate=ratio(daily_clean,daily_total),false_positive_facts=daily_fp),
        processing=dict(planned_cases=len(cases),saved_cases=len(rows),failed_cases=failed_cases),
        scope='Draft/pilot gold; manually reviewed one-to-one semantic matches. Missing/failed cases remain in denominators.')


def performance_metrics(report,rows):
    generations=[g for row in rows for g in row.get('generations',[])]
    ttfts=[g['ttft_ms'] for g in generations if g.get('ttft_ms') is not None]
    decode=[g for g in generations if g.get('decode_seconds') and g.get('decode_tokens') is not None]
    inputs=[g['input_tokens'] for g in generations if g.get('input_tokens') is not None]
    seconds=[row['seconds'] for row in rows]
    def gib(key):
        value=report.get(key)
        return value/2**30 if value is not None else None
    return dict(model=report['model'],parameters='4B (nominal)',
        precision='NF4 4-bit weights, BF16 compute' if report['device']=='cuda' else 'BF16',
        model_footprint_gib=gib('model_footprint_bytes'),model_cuda_allocated_gib=gib('model_cuda_allocated_bytes'),
        peak_allocated_gib=gib('peak_allocated_bytes'),peak_reserved_gib=gib('peak_reserved_bytes'),
        ttft_median_ms=statistics.median(ttfts) if ttfts else None,
        decode_tokens_per_second=ratio(sum(g['decode_tokens'] for g in decode),sum(g['decode_seconds'] for g in decode)),
        avg_input_tokens=statistics.mean(inputs) if inputs else None,
        warm_total_latency_median_seconds=statistics.median(seconds) if seconds and report.get('warmup_count',0)>report.get('warmup_failed',0) else None,
        total_latency_median_seconds=statistics.median(seconds) if seconds else None,
        total_latency_p95_seconds=sorted(seconds)[max(0,math.ceil(len(seconds)*0.95)-1)] if seconds else None,
        story_throughput_per_second=ratio(len(rows),sum(seconds)),generation_calls=len(generations),
        ttft_samples=len(ttfts),decode_samples=len(decode),input_token_samples=len(inputs),first_request_seconds=None,accuracy=None,
        measurement='Explicit full-pipeline warm-up excluded from measured rows; all measured failures and retries included. TTFT starts after tokenization. Decode excludes first token, includes EOS. VRAM PyTorch process counters, GiB, includes Qwen+BGE; reserved may include warm-up cache. First-request latency is not measured; model/encoder load reported separately. Accuracy pending reviewed semantic matches.')


def detection_markdown(metrics):
    def pct(value):return f'{value*100:.2f}%' if value is not None else 'N/A'
    lines=['## Detection metrics','', '| Metric | Precision | Recall | F1 |','|---|---|---|---|']
    for name,key in [('Fact extraction','extraction'),('Fact conflict detection','conflict'),('Story conflict detection','story_conflict')]:
        row=metrics[key];lines.append('| '+name+' | '+' | '.join(pct(row[x]) for x in ('precision','recall','f1'))+' |')
    lines+=['',f"Fact verdict accuracy: {pct(metrics['classification']['accuracy'])}",
            f"Core evidence Recall@{metrics['evidence']['k']}: {pct(metrics['evidence']['recall'])}",
            f"Clean empty extraction rate: {pct(metrics['daily']['clean_empty_rate'])}",'',metrics['scope']]
    return '\n'.join(lines)+'\n'
