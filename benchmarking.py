"""测完整编码吞吐，加载与质量评测独立于正式计时。"""
import gc
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parent


def summarize_times(samples,text_count):
    if not samples or text_count<1 or any(not math.isfinite(t) or t<=0 for t in samples):
        raise ValueError('耗时样本和文本数量必须有效且为正')
    median=statistics.median(samples)
    return {'samples_seconds':samples,'median_seconds':median,'texts_per_second':text_count/median}


def run_benchmark(index_directory,report_directory,repeats=3,warmups=1,threads=4,json_output=False):
    import numpy as np
    import torch
    import transformers
    from encoder import Encoder
    from evaluate import resolve_target,score_case
    from index_store import load_index,digest
    from retrieval import rank_chunks
    if any(isinstance(n,bool) or not isinstance(n,int) or n<1 for n in [repeats,warmups,threads]):
        raise ValueError('重复次数、预热次数和线程数必须为正整数')
    _,metadata=load_index(index_directory)
    texts=[c['embedding_text'] for c in metadata['chunks']]
    dataset_path=ROOT/'evaluation/queries.json'
    dataset=json.loads(dataset_path.read_text(encoding='utf-8'))
    cases=dataset['cases']
    targets={c['id']:(resolve_target(c['core'],metadata['chunks']),[resolve_target(t,metadata['chunks']) for t in c['additional']]) for c in cases}
    cuda=torch.cuda.is_available()
    original_threads=torch.get_num_threads()
    torch.set_num_threads(threads)
    variants=[]
    baseline=None
    baseline_orders=None
    print('测量使用同一索引快照；加载、预热、质量评测不计入编码耗时。',file=sys.stderr,flush=True)
    try:
        for device,precision in [('cpu','float32'),('cuda','float32'),('cuda','float16')]:
            if device=='cuda' and not cuda:
                variants.extend({'device':device,'precision':precision,'batch_size':batch,'status':'skipped','reason':'CUDA unavailable'} for batch in [1,16])
                continue
            print(f'加载 {device} {precision} 模型……',file=sys.stderr,flush=True)
            start=time.perf_counter()
            encoder=Encoder(device=device,precision=precision)
            if device=='cuda':
                torch.cuda.synchronize()
            load_seconds=time.perf_counter()-start
            expected=dict(metadata['config'])
            expected.pop('chunker',None)
            actual=dict(encoder.config,inference_dtype=metadata['config']['inference_dtype'])
            if actual!=expected:
                raise ValueError('索引与基准模型配置不匹配，请重建索引')
            try:
                for batch in [1,16]:
                    print(f'预热/计时 {device} {precision} batch={batch}……',file=sys.stderr,flush=True)
                    for _ in range(warmups):
                        encoder.encode_passages(texts,batch)
                    times=[]
                    peaks=[]
                    reserves=[]
                    baselines=[]
                    for _ in range(repeats):
                        if device=='cuda':
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            baselines.append(torch.cuda.memory_allocated())
                        start=time.perf_counter()
                        vectors=encoder.encode_passages(texts,batch)
                        if device=='cuda':
                            torch.cuda.synchronize()
                        times.append(time.perf_counter()-start)
                        if device=='cuda':
                            peaks.append(torch.cuda.max_memory_allocated())
                            reserves.append(torch.cuda.max_memory_reserved())
                    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5):
                        raise ValueError('基准输出不是有效单位向量')
                    queries=encoder.encode_queries([c['query'] for c in cases],batch)
                    results=[rank_chunks(vectors,metadata['chunks'],q,5) for q in queries]
                    orders=[[r['id'] for r in result] for result in results]
                    rows=[dict(id=c['id'],query=c['query'],**score_case(*targets[c['id']],result),top5=result) for c,result in zip(cases,results)]
                    if baseline is None:
                        baseline=(vectors.copy(),queries.copy())
                        baseline_orders=orders
                    expected_count=sum(len(r['expected_ids']) for r in rows)
                    missing=sum(len(r['missing_ids']) for r in rows)
                    variant=dict(device=device,precision=precision,batch_size=batch,status='passed',model_load_seconds=load_seconds,**summarize_times(times,len(texts)),peak_allocated_mib=max(peaks)/2**20 if peaks else None,peak_reserved_mib=max(reserves)/2**20 if reserves else None,baseline_allocated_mib=min(baselines)/2**20 if baselines else None,max_vector_absolute_difference=float(np.max(np.abs(vectors-baseline[0]))),mean_vector_cosine_to_baseline=float(np.mean(np.sum(vectors*baseline[0],axis=1))),max_query_absolute_difference=float(np.max(np.abs(queries-baseline[1]))),ordered_top5_identical_queries=sum(a==b for a,b in zip(orders,baseline_orders)),mean_top5_overlap=sum(len(set(a)&set(b))/5 for a,b in zip(orders,baseline_orders))/len(cases),core_hits=sum(r['core_hit'] for r in rows),complete_evidence_queries=sum(not r['missing_ids'] for r in rows),evidence_recall_micro=(expected_count-missing)/expected_count,query_results=rows)
                    variants.append(variant)
                    print(f"完成：{variant['median_seconds']:.4f}s，{variant['texts_per_second']:.1f} 片段/s，核心 {variant['core_hits']}/{len(cases)}",file=sys.stderr,flush=True)
            finally:
                del encoder
                gc.collect()
                if device=='cuda':
                    torch.cuda.empty_cache()
    finally:
        torch.set_num_threads(original_threads)
    report={'model_config':metadata['config'],'index_version':(Path(index_directory)/'CURRENT').read_text().strip(),'sources':metadata['sources'],'query_file_sha256':digest(dataset_path),'text_count':len(texts),'query_count':len(cases),'repeats':repeats,'warmup_full_corpus_rounds':warmups,'baseline':'cpu float32 batch=1','environment':{'platform':platform.platform(),'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'cpu_threads':threads,'original_cpu_threads':original_threads,'gpu':torch.cuda.get_device_name(0) if cuda else None},'variants':variants,'method':'计时涵盖输入长度检查、tokenization、传输、模型、FP32 L2 归一化与返回 NumPy；不含磁盘读写、模型加载、预热和质量评测。每种 device/precision 加载一次，两个 batch 共享该加载时间。GPU 每轮前后同步并重置峰值计数。','limitations':'48 个短测试片段；固定测量顺序，未控制笔记本功耗和温度。加载为本地缓存加载，不是冷启动下载。显存是本进程 PyTorch tensor/allocator 统计，不是整卡总显存。FP16 每配置重新编码资料与查询，正式 FP32 索引不修改。'}
    destination=Path(report_directory)
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 编码性能实测','',f'同一批 {len(texts)} 个片段，预热 {warmups} 轮，测量 {repeats} 轮，使用中位数。','', '| 配置 | batch | 全批耗时(s) | 片段/s | tensor峰值(MiB) | 核心命中 | Top-5顺序一致 |','|---|---:|---:|---:|---:|---:|---:|']
    for v in variants:
        if v['status']=='skipped':
            lines.append(f"| {v['device']} {v['precision']} | {v['batch_size']} | 跳过 | — | — | — | — |")
        else:
            peak=f"{v['peak_allocated_mib']:.1f}" if v['peak_allocated_mib'] is not None else '—'
            lines.append(f"| {v['device']} {v['precision']} | {v['batch_size']} | {v['median_seconds']:.4f} | {v['texts_per_second']:.1f} | {peak} | {v['core_hits']}/{len(cases)} | {v['ordered_top5_identical_queries']}/{len(cases)} |")
    lines+=['', 'Top-5 顺序与 CPU FP32 batch=1 比较；一致与否不等于证据找回率。', '',report['method'],'',report['limitations'],'','加载时间、每轮原始耗时、保留显存、向量差异与逐条排名见 benchmark.json。']
    (destination/'benchmark.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if json_output:
        print(json.dumps(report,ensure_ascii=False))
    else:
        print('\n'.join(lines))
    return report
