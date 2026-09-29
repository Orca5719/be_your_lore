"""General authoring flow: multiple suggestions, evidence, select and confirm."""
import argparse
import json
from pathlib import Path
import sys
from entries import EntryStore
from world_records import WorldStore, build_catalog, validate_plan
from entry import source_digests
ROOT=Path(__file__).resolve().parent
GENERATED='_world_records.md'
LABELS={'entity':'实体','attribute':'属性','relation':'关系','rule':'规则','event':'事件','timepoint':'时间节点'}


def prepare(store,legacy,chunks,text,judge,retriever,plan=None):
    value,initial_revision=store.read()
    old,_=legacy._read()
    records=[dict(row,file=str(store.path)) for row in value['records']]+[dict(row,file=str(legacy.path)) for row in old['entries']]
    catalog=build_catalog(chunks,records)
    extraction=judge.extract_world(text,catalog) if plan is None else dict(records=validate_plan(plan,text),prompt_version='author-edited-plan')
    if store.read()[1]!=initial_revision:
        raise ValueError('提取期间通用条目库已改变，请重新预览')
    preview=store.preview(extraction['records'],text,catalog)
    preview['extraction']=extraction
    if extraction.get('rejected'):
        preview['status']='partial_extraction'
    timing=extraction.get('timing',{})
    if timing:
        print(f"归档建议完成：{timing['seconds']:.1f} 秒；正在逐条核对证据…",file=sys.stderr,flush=True)
    checks=[]
    evidence={}
    for item in preview['items']:
        record=item['proposed']
        if record['kind'] in ('entity','timepoint'):
            item['review']=dict(status='ok',verdict='不确定',reason='实体/时间节点的归档建议需作者确认；名称已提及不等于类型或结构已建立',evidence=[])
            continue
        query=' '.join(filter(None,[record['entity'],record['target'],record['time'],record['text']]))
        results=retriever.search(query)
        for row in results:
            evidence.setdefault(row['id'],row)
        # A short, isolated claim prevents paragraph-level verdicts leaking across records.
        judgment=judge.check_world(record['text'],[dict(record,source_input=text)],results)
        checks.append(dict(text=record['text'],result=judgment))
        timing=judgment.get('timing',{})
        if timing:
            print(f"已核对第 {len(checks)} 条事实：{timing['seconds']:.1f} 秒",file=sys.stderr,flush=True)
        if judgment['status']!='ok':
            item['review']=dict(status='error',verdict=None,reason=judgment['error'],evidence=[])
        else:
            matched=[finding for finding in judgment['findings'] if record['text']==finding['input_quote']]
            if not matched:
                item['review']=dict(status='error',verdict=None,reason='模型未覆盖这条原文，不能据此保存',evidence=[])
            else:
                verdicts={row['verdict'] for row in matched}
                verdict='矛盾' if '矛盾' in verdicts else '不确定' if '不确定' in verdicts else '一致'
                item['review']=dict(status='ok',verdict=verdict,reason='；'.join(row['reason'] for row in matched),evidence=[citation for row in matched for citation in row['evidence']])
    preview['checks']=checks
    preview['judgment']={'evidence':list(evidence.values())}
    if any(item['review']['status']=='error' for item in preview['items']):
        preview['status']='check_failed'
    return preview


def display(preview):
    print('\n原输入：'+preview['source_input'])
    for rejected in preview['extraction'].get('rejected',[]):
        print('未处理的归档建议 ['+str(rejected['record_index'])+']：'+rejected['error'])
    if preview['extraction'].get('duplicates'):
        print('已合并完全相同的建议编号：'+str(preview['extraction']['duplicates']))
    for number,item in enumerate(preview['items'],1):
        record=item['proposed']
        print(f"\n[{number}] {LABELS[record['kind']]} | {record['entity'] or '全局'} → {record['category']}")
        if record['entity_type']:
            print('建议实体类型：'+record['entity_type'])
        if record['target']:
            print('关系对象：'+record['target'])
        print('拟记录原文：'+record['text'])
        print('事件时间：'+(record['time'] or '未指定')+'；生效：'+(record['valid_from'] or '未指定')+'；失效：'+(record['valid_until'] or '未指定'))
        for label,key in [('主体','entity_presence'),('对象','target_presence'),('时间','time_presence')]:
            if item[key]:
                print(label+'目录检查：'+item[key]['message'])
        print('已有同类条目：')
        for row in item['existing']:
            print(' - '+row['text']+' | '+str(row['source']))
        if not item['existing']:
            print(' （暂无）')
        print('判断：'+(item['review']['verdict'] or '检查失败')+'；'+item['review']['reason'])
        for citation in item['review']['evidence']:
            source=next(row for row in preview['judgment']['evidence'] if row['id']==citation['chunk_id'])
            print(' 证据：'+citation['quote']+' | '+source['file']+':'+str(source['start_line']))
    print('\n状态：'+('存在检查失败，失败条目不能保存' if preview['status']=='check_failed' else '部分提取成功；失败建议未处理，有效条目待确认，尚未保存' if preview['extraction'].get('rejected') else '全部待确认，尚未保存'))


def main(argv=None,session=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text',nargs='?')
    parser.add_argument('--interactive',action='store_true')
    parser.add_argument('--preview-only',action='store_true')
    parser.add_argument('--json',action='store_true')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--store',type=Path,default=ROOT/'data/world_records.json')
    parser.add_argument('--legacy-store',type=Path,default=ROOT/'data/entries.json')
    parser.add_argument('--lore',type=Path,default=ROOT/'lore')
    parser.add_argument('--index',type=Path,default=ROOT/'data/index')
    parser.add_argument('--plan',type=Path,help='人工修正的 {records:[...]} JSON，仍需原输入和证据检查')
    parser.add_argument('--sync-only',action='store_true',help='恢复已保存通用条目的导出与索引')
    args=parser.parse_args(argv)
    session={} if session is None else session
    try:
        if args.interactive:
            if args.plan:
                raise ValueError('连续模式不能重复使用同一人工计划')
            print('通用连续录入：逐行输入，/quit 退出，模型只加载一次。',file=sys.stderr,flush=True)
            options=['--store',str(args.store),'--legacy-store',str(args.legacy_store),'--lore',str(args.lore),'--index',str(args.index),'--device',args.device]
            options+=[flag for flag,enabled in [('--json',args.json),('--preview-only',args.preview_only)] if enabled]
            pending=args.text
            while True:
                if pending is None:
                    if sys.stdin.isatty():
                        print('新内容> ',end='',file=sys.stderr,flush=True)
                    line=sys.stdin.readline()
                    if not line:
                        return 0
                    text=line.strip()
                else:
                    text,pending=pending,None
                if text.lower() in ('/quit','/exit'):
                    return 0
                if text:
                    if main([text]+options,session)==130:
                        return 0
        from encoder import Encoder
        from index_store import build_index,load_index
        store=WorldStore(args.store)
        if args.sync_only:
            if not store.path.exists():
                raise ValueError('没有已保存的通用条目')
            store.export(args.lore/GENERATED)
            build_index(args.lore,args.index,Encoder(device='cpu'))
            print('已同步通用条目与索引')
            return 0
        if not args.text or not args.text.strip():
            raise ValueError('输入不能为空')
        _,metadata=load_index(args.index)
        if Path(metadata['lore_directory']).resolve()!=args.lore.resolve() or source_digests(args.lore)!=metadata['sources']:
            raise ValueError('资料目录/内容与索引不一致，请先重建')
        from retrieval import Retriever
        if 'encoder' not in session:
            session['encoder']=Encoder(device='cpu')
        retriever=Retriever(args.index,session['encoder'])
        retriever.search(args.text)  # Validate the short-query limit before loading Qwen.
        print('正在拆分归档建议与核对证据...',file=sys.stderr,flush=True)
        from qwen_judge import QwenJudge
        reused='judge' in session
        if not reused:
            session['judge']=QwenJudge(args.device)
        legacy=EntryStore(args.legacy_store)
        legacy_revision=legacy._read()[1]
        plan=json.loads(args.plan.read_text(encoding='utf-8-sig')) if args.plan else None
        chunks=[row for row in metadata['chunks'] if row['file'] not in (GENERATED,'_author_entries.md')]
        preview=prepare(store,legacy,chunks,args.text,session['judge'],retriever,plan)
        preview['session']={'models_reused':reused}
        print(json.dumps(preview,ensure_ascii=False),flush=True) if args.json else display(preview)
        if args.preview_only:
            return 0 if preview['status']=='pending_confirmation' else 2
        print('确认全部输入“确认保存”；只选部分输入“确认保存 1,3”；其他输入取消：',file=sys.stderr,flush=True)
        response=sys.stdin.readline().strip()
        if response=='确认保存':
            if preview['extraction'].get('rejected'):
                raise ValueError('部分提取失败，请明确选择有效条目编号，例如“确认保存 1,3”；未保存')
            selected=list(range(len(preview['items'])))
        elif response.startswith('确认保存 '):
            try:
                selected=[int(part.strip())-1 for part in response[5:].replace('，',',').split(',')]
            except ValueError as exc:
                raise ValueError('选择编号无效，未保存') from exc
        else:
            print(json.dumps({'status':'cancelled'},ensure_ascii=False) if args.json else '已取消，未保存')
            return 0
        if not selected or any(i<0 or i>=len(preview['items']) or preview['items'][i]['review']['status']!='ok' for i in selected):
            raise ValueError('选中了失败或无效条目，未保存')
        if source_digests(args.lore)!=metadata['sources'] or legacy._read()[1]!=legacy_revision:
            raise ValueError('预览后已有资料改变，请重新预览')
        saved=store.commit(preview,selected)
        status={'status':'saved','count':len(saved),'records':saved,'index_status':'pending'}
        try:
            store.export(args.lore/GENERATED)
            version=build_index(args.lore,args.index,session['encoder'])
            status.update(index_status='ready',index_version=version.name)
        except (ValueError,OSError) as exc:
            status.update(index_status='error',index_error=str(exc))
        print(json.dumps(status,ensure_ascii=False),flush=True) if args.json else print('已确认保存 '+str(len(saved))+' 条；检索同步：'+status['index_status'])
        if status['index_status']=='error':
            print('条目已保存；同步失败请用 --sync-only 恢复：'+status['index_error'],file=sys.stderr)
        return 0 if status['index_status']=='ready' else 2
    except (ValueError,OSError,RuntimeError) as exc:
        if args.json:
            print(json.dumps(dict(status='error',error=str(exc),raw_output=getattr(exc,'raw_output',None)),ensure_ascii=False),flush=True)
        print('错误：'+str(exc),file=sys.stderr,flush=True)
        return 2
    except KeyboardInterrupt:
        print('已退出；若已确认请检查条目库。',file=sys.stderr)
        return 130

if __name__=='__main__':
    raise SystemExit(main())




