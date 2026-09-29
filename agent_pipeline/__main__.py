import argparse
import json
import sys
from pathlib import Path
from .understanding import understand
from .filtering import filter_events,validate_input
from .retrieval import retrieve_events,validate_filtering
from .judge import judge_events,validate_retrieval
from .report import build_report,validate_judge,render_author_report
from .story_check import check_story


def main():
    parser=argparse.ArgumentParser(description='独立agent_pipeline：理解、筛选、检索与判断')
    parser.add_argument('command',choices=['understand','filter','retrieve','judge','report'])
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--text');source.add_argument('--file',type=Path)
    source.add_argument('--events-file',type=Path,help='filter专用：第一模块完整JSON报告，不重新理解')
    source.add_argument('--filter-file',type=Path,help='retrieve专用：筛选完整报告，不调用LLM')
    source.add_argument('--judge-file',type=Path,help='report专用：完整judge报告，只整理，不加载模型')
    source.add_argument('--retrieval-file',type=Path,help='judge专用：检索完整JSON报告，跳过前三层')
    parser.add_argument('--index',type=Path,help='retrieve使用的已有索引；默认项目data/index')
    parser.add_argument('--top-k',type=int,default=5)
    parser.add_argument('--retrieval-device',choices=['auto','cpu','cuda'],help='独立设置embedding设备；默认跟随device')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--output',type=Path,help='保存完整JSON报告；不覆盖已有文件')
    parser.add_argument('--compact',action='store_true',help='终端只显示事件与处理摘要；文件仍保存完整记录')
    parser.add_argument('--json',action='store_true',help='report终端输出JSON；默认输出面向作者的中文报告')
    args=parser.parse_args()
    try:
        if args.output and args.output.exists():raise ValueError('输出文件已存在；请换路径')
        llm=None
        if args.events_file and args.command not in ('filter','retrieve','judge','report'):raise ValueError('--events-file仅用于filter/retrieve/judge命令')
        if args.filter_file and args.command not in ('retrieve','judge','report'):raise ValueError('--filter-file仅用于retrieve/judge命令')
        if args.retrieval_file and args.command not in ('judge','report'):raise ValueError('--retrieval-file仅用于judge命令')
        if args.command in ('retrieve','judge','report') and args.top_k<1:raise ValueError('top-k必须为正整数')
        def progress(step):
            label='检查故事内事件' if step['purpose']=='story_check' else '核对设定' if step['purpose']=='judge' else '检索相关设定' if step['purpose']=='retrieval' else '筛选事件' if step['purpose']=='filtering' else '补提覆盖缺口' if step['purpose']=='coverage_recovery' else '理解原文'
            print(f'窗口 {step["window_id"]}：{label}',file=sys.stderr,flush=True)
        if args.judge_file:
            if args.command!='report':raise ValueError('--judge-file仅用于report命令')
            result=json.loads(args.judge_file.read_text(encoding='utf-8-sig'))
            validate_judge(result)
        elif args.retrieval_file:
            result=json.loads(args.retrieval_file.read_text(encoding='utf-8-sig'))
            validate_retrieval(result)
        elif args.filter_file:
            result=json.loads(args.filter_file.read_text(encoding='utf-8-sig'))
            validate_filtering(result)
        elif args.events_file:
            upstream=json.loads(args.events_file.read_text(encoding='utf-8-sig'))
            validate_input(upstream)
            if args.command in ('judge','report') and upstream['events']:
                from qwen_judge import QwenJudge
                llm=QwenJudge(args.device)
            result=filter_events(upstream,device=args.device,llm=llm,progress=progress)
        else:
            text=args.text if args.text is not None else args.file.read_text(encoding='utf-8-sig')
            if args.command=='understand':
                print('正在本地理解全部语义事件（不筛选、不检索、不判断）…',file=sys.stderr)
                result=understand(text,device=args.device,progress=progress)
            else:
                if not text.strip() or len(text)>800:raise ValueError('故事必须非空且不超过800字符；未截断')
                from qwen_judge import QwenJudge
                print('正在本地理解、筛选、检索并核对设定（LLM阶段共享模型）…' if args.command in ('judge','report') else '正在本地理解并筛选事件（两阶段共享模型，不判断矛盾）…',file=sys.stderr)
                llm=QwenJudge(args.device)
                upstream=understand(text,llm=llm,progress=progress)
                result=filter_events(upstream,llm=llm,progress=progress)
                result['model_loaded_this_request']=True
        if args.command in ('retrieve','judge','report') and not args.retrieval_file and not args.judge_file:
            result=retrieve_events(result,device=args.retrieval_device or args.device,index_directory=args.index,k=args.top_k,progress=progress)
        if args.command in ('judge','report') and not args.judge_file:
            result=judge_events(result,device=args.device,llm=llm,progress=progress)
            filtering=result['retrieval']['filtering']
            if filtering is not None:
                result['story_check']=check_story(filtering,llm=llm,device=args.device,progress=progress)
                if result['story_check']['status']!='ok':
                    if result['status']=='ok':result['status']='partial'
                    result['processing_complete']=False
                    result['failure_reasons']['story_check']='段内检查失败或未完整完成，详见story_check'
        if args.command=='report':result=build_report(result)
        rendered=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
        if args.output:
            with args.output.open('x',encoding='utf-8') as stream:stream.write(rendered)
        if args.command=='report' and not args.json:
            sys.stdout.write(render_author_report(result))
            if args.output:print('完整 JSON 已保存：'+str(args.output.resolve()))
            else:print('本次未保存 JSON；可用 --output 指定保存路径。')
        elif args.compact:
            display={key:result[key] for key in ['schema_version','stage','status','events','notice']}
            event_keys=['id','actors','event','mental_state','explicit','modality','conditions','source_ids','context_ids','review_required','review_reasons']
            display['events']=[{key:event[key] for key in event_keys if key in event} for event in result['events']]
            if args.command=='report':
                display['report']=result['report']
                for key in ('summary','evidence','review_items','failure_reasons','processing_complete'):display[key]=result[key]
                display.pop('events')
            elif args.command=='judge':
                display['items']=[{key:item[key] for key in ('event_id','status','verdict','label','citations','reason','origin','assessment','model_verdict','required_by_event_ids','error') if key in item} for item in result['items']]
                for key in ('pending_event_ids','ignored_event_ids','upstream_status','failure_reasons','processing_complete'):display[key]=result[key]
                if 'story_check' in result:display['story_check']=result['story_check']
            elif args.command=='retrieve':
                display['items']=[{key:item[key] for key in ('event_id','status','query','evidence','required_by_event_ids','error') if key in item} for item in result['items']]
                for key in ('pending_event_ids','ignored_event_ids','upstream_status','failure_reasons','processing_complete','top_k'):display[key]=result[key]
            elif args.command=='filter':
                display['decisions']=result['decisions']
                display['selected_event_ids']=[e['id'] for e in result['selected_events']]
                display['pending_event_ids']=[e['id'] for e in result['pending_events']]
                display['upstream_status']=result['upstream_status']
                display['failure_reasons']=result.get('failure_reasons',{})
                display['processing_complete']=result.get('processing_complete',False)
            else:
                display['coverage']=dict(missing_source_ids=result['coverage']['missing_source_ids'],note=result['coverage']['note'])
                display['failure_reasons']=result.get('failure_reasons',{})
            display['rejected_count']=len(result.get('rejected',[]))
            if args.command in ('judge','report'):
                judged=result['judge'] if args.command=='report' else result
                understanding=judged['retrieval']['filtering'].get('understanding',{})
                display['upstream_rejected_count']=len(understanding.get('rejected',[]))
                display['unresolved_upstream_rejected_count']=len(understanding.get('failure_reasons',{}).get('unresolved_candidates',[]))
            display['semantic_verification']=result.get('semantic_verification','not_verified')
            display['review_required']=result.get('review_required',True)
            sys.stdout.write(json.dumps(display,ensure_ascii=False,indent=2)+'\n')
        else:sys.stdout.write(rendered)
        if result['status']!='ok':print('本次未完整通过：'+result['status']+'；退出码2。请查看failure_reasons及完整报告。',file=sys.stderr)
        return 0 if result['status']=='ok' else 2
    except (ValueError,OSError,RuntimeError) as exc:
        print(json.dumps({'stage':{'understand':'understanding','filter':'filtering','retrieve':'retrieval','judge':'judge','report':'report'}[args.command],'status':'error','error':str(exc)},ensure_ascii=False))
        return 2


if __name__=='__main__':raise SystemExit(main())
