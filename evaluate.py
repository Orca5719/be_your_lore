"""固定人工标签的 Top-5 检索评测；不评价一致性推理。"""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def score_case(core_id,additional_ids,results):
    ids=[r['id'] for r in results]
    expected=[core_id,*additional_ids]
    missing=[id for id in expected if id not in ids]
    return {'core_hit':core_id in ids,'core_rank':ids.index(core_id)+1 if core_id in ids else None,'expected_ids':expected,'missing_ids':missing,'evidence_recall':(len(expected)-len(missing))/len(expected)}


def resolve_target(target,chunks):
    matches=[c['id'] for c in chunks if c['file']==target['file'] and c['heading_path'][-len(target['heading_suffix']):]==target['heading_suffix']]
    if len(matches)!=1:
        raise ValueError(f'标签必须唯一匹配一个片段，实际 {len(matches)}：{target}')
    return matches[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild',action='store_true',help='先按当前资料重建正式索引')
    args=parser.parse_args()
    from encoder import Encoder
    from index_store import build_index,digest
    from retrieval import Retriever
    dataset_path=ROOT/'evaluation/queries.json'
    dataset=json.loads(dataset_path.read_text(encoding='utf-8'))
    if len(dataset['cases'])!=20 or dataset['k']!=5 or len({c['id'] for c in dataset['cases']})!=20:
        raise ValueError('阶段验收要求固定 20 条唯一查询，K=5')
    encoder=Encoder()
    if args.rebuild:
        build_index(ROOT/'lore',ROOT/'data/index',encoder)
    retriever=Retriever(ROOT/'data/index',encoder)
    metadata=retriever.metadata
    sources={p.relative_to(ROOT/'lore').as_posix():digest(p) for p in sorted((ROOT/'lore').rglob('*')) if p.is_file() and p.suffix.lower() in {'.md','.txt'}}
    if sources!=metadata['sources']:
        raise ValueError('索引与当前资料不同，请使用 --rebuild 重建')
    if len(metadata['chunks'])<40:
        raise ValueError('阶段验收需要至少 40 个片段')
    # 在检索之前解析全部标签，缺失标签不会计作检索失败或被忽略。
    targets={case['id']:(resolve_target(case['core'],metadata['chunks']),[resolve_target(t,metadata['chunks']) for t in case['additional']]) for case in dataset['cases']}
    rows=[]
    for case in dataset['cases']:
        results=retriever.search(case['query'],k=dataset['k'])
        core,additional=targets[case['id']]
        row=dict(case,**score_case(core,additional,results),results=results)
        rows.append(row)
        print(f"{case['id']} 核心排名={row['core_rank']} 证据找回={row['evidence_recall']:.0%} | {case['query']}")
    hits=sum(row['core_hit'] for row in rows)
    expected=sum(len(row['expected_ids']) for row in rows)
    missing=sum(len(row['missing_ids']) for row in rows)
    core_example=next(row for row in rows if row['id']=='q01')
    summary={'queries':len(rows),'chunks':len(metadata['chunks']),'k':5,'core_hits':hits,'core_hit_rate':hits/len(rows),'all_evidence_complete_queries':sum(not row['missing_ids'] for row in rows),'evidence_recall_micro':(expected-missing)/expected,'expected_evidence_count':expected,'missing_evidence_count':missing,'core_example_passed':core_example['core_hit'],'acceptance_passed':hits>=16 and core_example['core_hit']}
    report={'summary':summary,'model_config':metadata['config'],'index_version':(ROOT/'data/index/CURRENT').read_text().strip(),'sources':sources,'query_file_sha256':digest(dataset_path),'cases':rows,'unrelated_diagnostic':{'query':'番茄炒蛋需要放多少盐？','results':retriever.search('番茄炒蛋需要放多少盐？')},'limitations':'固定小型合成测试库；不代表真实作品上的泛化质量。不判断矛盾。标签在模型检索前固定，本次未按结果修改标签。'}
    reports=ROOT/'reports'
    reports.mkdir(exist_ok=True)
    (reports/'retrieval_evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 世界观检索评测（测试资料）','',f"- 资料：{summary['chunks']} 个片段，20 条预先标注查询，Top-5。",f'- 核心证据命中：{hits}/20。',f"- 完整找回全部证据：{summary['all_evidence_complete_queries']}/20。",f"- 标注证据总找回率：{summary['evidence_recall_micro']:.1%}（{expected-missing}/{expected}）。",f"- 核心左胸示例：{'通过' if summary['core_example_passed'] else '未通过'}。",f"- 阶段检索验收：{'通过' if summary['acceptance_passed'] else '未通过'}。",'','| 查询 | 核心排名 | 证据找回率 |','|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['id']} {row['query']} | {row['core_rank'] or '未找回'} | {row['evidence_recall']:.0%} |")
    lines+=['','## 漏掉的证据','']
    for row in rows:
        if row['missing_ids']:
            names=[' > '.join(c['heading_path']) for c in metadata['chunks'] if c['id'] in row['missing_ids']]
            lines.append(f"- {row['id']}："+'；'.join(names))
    if not missing:
        lines.append('无。')
    lines+=['',report['limitations'],'','完整来源、分数、标签、摘要与诊断结果见 retrieval_evaluation.json。']
    (reports/'retrieval_evaluation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0 if summary['acceptance_passed'] else 1

if __name__=='__main__':
    raise SystemExit(main())
