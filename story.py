"""Story-Level Consistency Benchmark: extraction and per-fact retrieval CLI."""
import argparse
import json
from pathlib import Path
import sys
from story_extraction import extract_story

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['extract','retrieve','check'])
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text')
    group.add_argument('--file',type=Path)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--json',action='store_true')
    parser.add_argument('--index',type=Path,default=Path(__file__).resolve().parent/'data/index')
    parser.add_argument('--top-k',type=int,default=5)
    args=parser.parse_args(argv)
    try:
        text=args.text if args.text is not None else args.file.read_text(encoding='utf-8-sig')
        print('正在本地提取、检索并逐事实判断…' if args.command=='check' else '正在本地提取故事中的重要事实'+('并逐条检索证据' if args.command=='retrieve' else '')+'（本步骤不判断矛盾）…',file=sys.stderr,flush=True)
        if args.command=='check':
            from story_judgement import check_story
            result=check_story(text,args.device,args.index,args.top_k)
        elif args.command=='retrieve':
            from story_retrieval import retrieve_story
            result=retrieve_story(text,args.device,args.index,args.top_k)
        else:result=extract_story(text,args.device)
        if args.json:print(json.dumps(result,ensure_ascii=False))
        else:
            print(result['notice'])
            extraction=result['extraction'] if args.command!='extract' else result
            for fact in extraction['facts']:
                print('\n'+fact['id']+' ['+fact['type']+'] '+fact['subject']+' → '+fact['predicate']+' → '+(fact['object'] or '（无独立对象）'))
                print('原文：'+fact['source_text'])
                if fact['context_text']:print('必要上下文：'+fact['context_text'])
                if fact['time']:print('时间：'+fact['time'])
                if args.command!='extract':
                    item=next(x for x in result['items'] if x['fact']['id']==fact['id'])
                    if item['status']=='error':print('检索失败：'+item['error'])
                    elif not item['evidence']:print('没有返回候选证据。')
                    for rank,evidence in enumerate(item['evidence'],1):
                        print(str(rank)+'. ['+evidence['id']+'] cosine='+format(evidence['score'],'.4f'))
                        print(evidence['text'])
                        print('来源：'+evidence['file']+' 行 '+str(evidence['start_line'])+'-'+str(evidence['end_line'])+' 标题：'+' / '.join(evidence.get('heading_path',[])))
                    if args.command=='check':
                        answer=item['judgement']
                        if answer['status']=='error':print('判断失败：'+answer['error'])
                        else:
                            print('判断：'+answer['verdict']+'；原因：'+answer['reason'])
                            print('引用证据：'+(', '.join(answer['evidence_ids']) or '无'))
            if not extraction['facts'] and extraction['status']=='ok':print('未提取到需要检查的重要事实。')
            for fact in extraction.get('pending_facts',[]):
                print('待复核候选 '+fact['id']+'：'+fact['subject']+' → '+fact['predicate']+' → '+(fact['object'] or '（无独立对象）'))
                print('原文：'+fact['source_text'])
                if fact['context_text']:print('必要上下文：'+fact['context_text'])
            for issue in extraction.get('review_issues',[]):
                print('筛选问题 '+issue['id']+'：'+issue['error'])
            for row in extraction.get('discarded',[]):
                print('忽略候选 '+row['id']+'：'+row['reason'])
            for row in extraction['rejected']:print('拒绝候选 '+row['id']+'：'+row['error'])
            if extraction.get('error'):print('错误：'+extraction['error'])
            if args.command=='check':
                print('\n汇总：'+result['summary']['display'])
                print(result['summary']['scope'])
            print('\n处理状态：'+result['status'])
        return 0 if result['status']=='ok' else 2
    except (ValueError,OSError,RuntimeError) as exc:
        if args.json:print(json.dumps({'status':'error','error':str(exc)},ensure_ascii=False))
        print('错误：'+str(exc),file=sys.stderr)
        return 2
    except KeyboardInterrupt:return 130

if __name__=='__main__':raise SystemExit(main())
