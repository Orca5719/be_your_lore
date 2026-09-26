"""录入预览 -> 判断 -> 作者确认 -> 保存 -> 同步检索索引。"""
import argparse
import json
from pathlib import Path
import sys
from entries import EntryStore

ROOT = Path(__file__).resolve().parent
GENERATED = '_author_entries.md'


def source_digests(root):
    import hashlib
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.suffix.lower() in {'.md','.txt'}}


def continuous(args, session):
    print('连续录入：首次加载模型，后续复用。输入 /quit 退出。',file=sys.stderr,flush=True)
    options = ['--store',str(args.store),'--lore',str(args.lore),'--index',str(args.index),'--device',args.device]
    for flag,value in [('--entity',args.entity),('--category',args.category),('--time',args.time)]:
        if value is not None:
            options += [flag,value]
    for flag,enabled in [('--preview-only',args.preview_only),('--json',args.json)]:
        if enabled:
            options.append(flag)
    pending = args.text
    try:
        while True:
            if pending is None:
                if sys.stdin.isatty():
                    print('新设定> ',end='',file=sys.stderr,flush=True)
                line = sys.stdin.readline()
                if not line:
                    return 0
                text = line.strip()
            else:
                text,pending = pending,None
            if text.lower() in ('/quit','/exit'):
                return 0
            if not text.strip():
                continue
            code = main([text]+options,session=session)
            if code == 130:
                return 0
    except KeyboardInterrupt:
        print('已退出。',file=sys.stderr)
        return 0


def main(argv=None,session=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text',nargs='?',help='拟新增设定原文；连续模式可不提供')
    parser.add_argument('--world',action='store_true',help='通用多条录入：实体、关系、规则、事件与时间')
    parser.add_argument('--plan',type=Path,help='通用模式人工修正计划 JSON')
    parser.add_argument('--sync-only',action='store_true',help='通用模式恢复已保存条目的索引同步')
    parser.add_argument('--interactive',action='store_true',help='连续录入，模型只加载一次')
    parser.add_argument('--entity', help='可选：覆盖自动识别的人物/实体，例如雷')
    parser.add_argument('--category', help='可选：覆盖自动分类，例如能力、生理结构、经历、timeline')
    parser.add_argument('--time', help='可选：覆盖自动识别的原文时间；不会补写未出现的日期')
    parser.add_argument('--store', type=Path, default=ROOT/'data/entries.json')
    parser.add_argument('--lore', type=Path, default=ROOT/'lore')
    parser.add_argument('--index', type=Path, default=ROOT/'data/index')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto', help='Qwen 设备，检索和重建使用 CPU FP32')
    parser.add_argument('--preview-only', action='store_true', help='只预览和判断，不询问保存')
    parser.add_argument('--json', action='store_true', help='预览和保存结果以 JSON 分行输出')
    args = parser.parse_args(argv)
    if args.world:
        if args.entity is not None or args.category is not None or args.time is not None:
            parser.error('通用多条模式请在预览后用 --plan 修正，不能把整段强制归于一个人物/类别')
        from world_entry import main as world_main
        options=['--lore',str(args.lore),'--index',str(args.index),'--device',args.device]
        if args.store != ROOT/'data/entries.json':
            options+=['--store',str(args.store)]
        if args.text is not None:
            options=[args.text]+options
        for flag,enabled in [('--interactive',args.interactive),('--preview-only',args.preview_only),('--json',args.json),('--sync-only',args.sync_only)]:
            if enabled:
                options.append(flag)
        if args.plan:
            options+=['--plan',str(args.plan)]
        return world_main(options,session)
    if args.plan or args.sync_only:
        parser.error('--plan / --sync-only 需要 --world')
    session = {} if session is None else session
    if args.interactive:
        return continuous(args,session)
    if args.text is None:
        parser.error('请提供设定，或使用 --interactive 连续录入')
    try:
        store = EntryStore(args.store)
        store._validate(args.entity if args.entity is not None else '待识别',args.category if args.category is not None else '待识别',args.text,args.time)
        from index_store import load_index, build_index
        from encoder import Encoder
        from retrieval import Retriever
        _, metadata = load_index(args.index)
        if Path(metadata['lore_directory']).resolve() != args.lore.resolve():
            raise ValueError('资料目录与索引不匹配，请重建')
        if source_digests(args.lore) != metadata['sources']:
            raise ValueError('资料已修改，先重建索引后再录入')
        chunks = [x for x in metadata['chunks'] if x['file'] != GENERATED]
        models_reused = 'judge' in session
        if 'encoder' not in session:
            session['encoder'] = Encoder(device='cpu',offline=True)
        encoder = session['encoder']
        retriever = Retriever(args.index,encoder)
        judgment_text = ('事件时间：' + args.time + '\n' if args.time else '') + args.text
        results = retriever.search(judgment_text)
        print('正在本地识别与判断...',file=sys.stderr,flush=True)
        from qwen_judge import QwenJudge
        if 'judge' not in session:
            session['judge'] = QwenJudge(args.device)
        judge = session['judge']
        classification = dict(status='manual',entity=args.entity,category=args.category,event_time=args.time)
        if args.entity is None or args.category is None:
            authored, _ = store._read()
            entities = sorted({x['heading_path'][-2] for x in chunks if len(x['heading_path']) >= 2} | {x['entity'] for x in authored['entries']})
            categories = sorted({'能力','生理结构','经历','身份','关系','装备','规则','timeline'} | {x['heading_path'][-1] for x in chunks if len(x['heading_path']) >= 2} | {x['category'] for x in authored['entries']})
            classification = judge.classify(args.text,entities,categories,overrides={'entity':args.entity,'category':args.category,'event_time':args.time})
            if classification['status'] == 'needs_clarification':
                if args.json:
                    print(json.dumps(dict(status='needs_clarification',classification=classification),ensure_ascii=False),flush=True)
                if args.preview_only or not sys.stdin.isatty():
                    raise ValueError('实体或类别不明确，请补充 --entity / --category；未保存')
                for field in classification['missing']:
                    print('请补充'+('人物/实体' if field=='entity' else '类别')+'：',file=sys.stderr,flush=True)
                    classification[field] = sys.stdin.readline().strip()
                classification['status'] = 'author_corrected'
            args.entity = classification['entity']
            args.category = classification['category']
            args.time = classification['event_time']
        store._validate(args.entity,args.category,args.text,args.time)
        preview = store.preview(args.entity,args.category,args.text,chunks,event_time=args.time)
        preview['classification'] = classification
        preview['session'] = dict(models_reused=models_reused,qwen_load_seconds=0.0 if models_reused else getattr(judge,'load_seconds',None))
        judgment_text = ('事件时间：' + args.time + '\n' if args.time else '') + args.text
        results = retriever.search(judgment_text)
        preview['judgment'] = judge.check(judgment_text,results)
        if preview['judgment']['status'] != 'ok':
            preview['status'] = 'check_failed'
        if args.json:
            print(json.dumps(preview,ensure_ascii=False),flush=True)
        else:
            print(f"\n实体：{preview['entity']}　类别：{preview['category']}")
            print('已有同类条目：')
            for item in preview['existing']:
                print(' -',item['text'],'| 来源：',item['source'])
            if not preview['existing']:
                print(' （暂无已记录条目）')
            print('其他类别：')
            for category, items in preview['other_categories'].items():
                for item in items:
                    print(f" - 【{category}】{item['text']}")
            print('拟新增：',preview['proposed']['text'])
            print('事件时间：',args.time or '未指定')
            if models_reused:
                print('模型：复用已加载模型')
            elif getattr(judge,'load_seconds',None) is not None:
                print(f'首次模型加载：{judge.load_seconds:.2f} 秒；连续模式后续省去此步骤。')
            print('判断：',preview['judgment']['overall_verdict'] or '检查失败')
            for finding in preview['judgment'].get('findings',[]):
                print(' -',finding['verdict'],finding['reason'])
                for citation in finding['evidence']:
                    source = next(x for x in results if x['id'] == citation['chunk_id'])
                    print('   证据：',citation['quote'], '|', source['file'], str(source['start_line'])+'-'+str(source['end_line']))
            if preview['judgment']['status'] != 'ok':
                print('失败原因：',preview['judgment']['error'])
                print('状态：检查失败，未保存。这不是矛盾判断；可重新检查。')
            else:
                print('状态：待确认；不确定表示当前证据不足，作者确认后可建立新设定。')
            classification_seconds = classification.get('timing',{}).get('seconds')
            check_seconds = preview['judgment'].get('timing',{}).get('seconds')
            if check_seconds is not None:
                print('生成耗时：识别 '+(f'{classification_seconds:.2f} 秒' if classification_seconds is not None else '手动')+f'，检查 {check_seconds:.2f} 秒。')
        if args.preview_only:
            return 0 if preview['judgment']['status']=='ok' else 2
        if preview['judgment']['status'] != 'ok':
            raise ValueError('模型检查失败，本次不允许保存：' + preview['judgment']['error'])
        print('人物、类别和时间正确，并确认将拟新增原文保存为作者设定？输入“确认保存”，其他输入取消：',file=sys.stderr,flush=True)
        if sys.stdin.readline().strip() != '确认保存':
            print(json.dumps({'status':'cancelled'},ensure_ascii=False) if args.json else '已取消，未写入。')
            return 0
        if source_digests(args.lore) != metadata['sources']:
            raise ValueError('判断后资料改变，请重新预览')
        entry = store.commit(preview)
        status = dict(status='saved',message='新设定已记录',entry=entry,index_status='pending')
        try:
            store.export(args.lore/GENERATED)
            version = build_index(args.lore,args.index,encoder)
            status.update(index_status='ready',index_version=version.name)
        except (ValueError,OSError) as exc:
            status.update(index_status='error',index_error=str(exc))
        if args.json:
            print(json.dumps(status,ensure_ascii=False),flush=True)
        else:
            print('新设定已记录。已有【'+args.category+'】设定：')
            for item in store.preview(args.entity,args.category,args.text,chunks,event_time=args.time)['existing']:
                print(' -',item['text'])
            print('检索索引：',status['index_status'])
            if 'index_error' in status:
                print('保存成功但同步失败：',status['index_error'],'；请运行 sync_entries.py 恢复同步。')
        return 0 if status['index_status']=='ready' else 2
    except (ValueError,OSError) as exc:
        print('错误：'+str(exc),file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('已退出；若已确认保存，请检查 entries.json。',file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())







