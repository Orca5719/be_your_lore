"""Deterministic, traceable presentation of existing judge results. No inference."""
import copy
import json
from .judge import LABELS,validate_retrieval
from .retrieval import validate_filtering

NOTICE='仅汇总已核对事件，不证明整段故事无矛盾；不确定不是处理失败。保留上游建议，未保存设定。相似度不是事实正确率或矛盾置信度。'


def presentation_report(findings,counts,status,failed,pending,unprocessed,filtering,failure_reasons):
    """Compose a complete feedback document without adding inferred story facts."""
    sections={'contradictions':[],'uncertainties':[],'confirmations':[]}
    names={'contradiction':'contradictions','uncertain':'uncertainties','consistent':'confirmations'}
    topics=[];additions=[]
    events={e['id']:e for e in filtering['events']}
    for finding in findings:
        entry=dict(finding_ids=[finding['id']],event_ids=finding['event_ids'],actors=finding['actors'],fact=finding['event'],reasons=finding['reasons'],citations=finding['citations'],reason_origin=finding.get('reason_origin'),uncertainty_code=finding.get('uncertainty_code'))
        sections[names[finding['verdict']]].append(entry)
        if finding['verdict']=='uncertain':
            if all(events[eid]['modality'] in ('observed','conditional') for eid in finding['event_ids']) and finding.get('origin')!='program_nonactual_scope' and (finding.get('uncertainty_code')=='missing_rule' or finding.get('origin')=='program_no_evidence'):
                additions.append(dict(**entry,kind='unclassified',saved=False,notice='本次检索未覆盖这一事实；不能据此认定全库没有记录。请确认是新增设定、剧情事实，还是检索遗漏；一次行为不自动成为永久能力。'))
            question='请核实适用设定及检索/判断结果，再决定是否建立新设定；不确定不等于新增设定。'
            if finding.get('origin')=='program_nonactual_scope':question='这是非客观叙述建议，请核实叙述性质及其对应约束，不能直接建立现实设定。'
            topics.append(dict(finding_ids=[finding['id']],event_ids=finding['event_ids'],fact=finding['event'],question=question,saved=False))
    checked=sum(counts[k] for k in LABELS)
    overview=f"已核对{checked}项重要事实：{counts['contradiction']}项明确矛盾、{counts['uncertain']}项证据不足或无法明确判定、{counts['consistent']}项明确吻合。"
    if not checked:overview='没有成功核对的实质事实，不能据此认定故事一致。'
    if status!='ok':overview+='本次处理未完整完成，见processing_issues。'
    if counts['ignored'] or counts['support_only']:overview+=f"另有{counts['ignored']}项已过滤、{counts['support_only']}项仅保留为上下文，不独立核对。"
    overview+='结论仅覆盖已提取并核对的事实，未保存设定。'
    events={e['id']:e for e in filtering['events']}
    exclusions=[dict(event_id=d['event_id'],actors=events[d['event_id']]['actors'],fact=events[d['event_id']]['event'],reason=d['reason'],role='context_only' if d.get('support_only') else 'ignored',required_by_event_ids=d.get('required_by_event_ids',[])) for d in filtering['decisions'] if d['decision']=='ignore' or d.get('support_only')]
    processing=[]
    for role,items in [('failed',failed),('pending',pending),('unprocessed',unprocessed)]:
        for item in items:
            event=item['event'] if isinstance(item.get('event'),dict) else item
            processing.append(dict(kind=role,event_id=item.get('event_id',item.get('id')),actors=event['actors'],fact=event['event'],detail=item.get('error') or item.get('reason') or '需要复核或尚未核对'))
    if status!='ok':processing.append(dict(kind='upstream_incomplete',details=failure_reasons))
    return dict(overview=overview,scope='checked_events',**sections,possible_additions=additions,confirmation_topics=topics,excluded_details=exclusions,processing_issues=processing)


def render_author_report(result):
    """Present existing decisions to an author; never infer or save new canon."""
    report=result['report'];lines=['故事检查报告',report['overview']]
    findings={f['id']:f for f in result['findings']}
    evidence={e['id']:e for e in result['evidence']}
    additions=report['possible_additions']
    candidate_ids={fid for entry in additions for fid in entry['finding_ids']}
    sections=[('与已有内容矛盾',report['contradictions']),('可能新增的设定或剧情事实（待确认，未保存）',additions),('需要确认',[entry for entry in report['uncertainties'] if not candidate_ids.intersection(entry['finding_ids'])]),('与已有设定明确吻合',report['confirmations'])]
    for title,entries in sections:
        lines.extend(['',title+'：'])
        if not entries:lines.append('无。')
        for number,entry in enumerate(entries,1):
            lines.append(f"{number}. {entry['fact']}")
            for fid in entry['finding_ids']:
                finding=findings[fid]
                for source in finding['sources']:lines.append('   原文：'+source['text'])
                for context in finding['contexts']:lines.append('   上下文：'+context['text'])
            for reason in entry['reasons']:lines.append('   原因：'+reason)
            for citation in entry['citations']:
                chunk=evidence[citation['chunk_id']]
                lines.append(f"   已有记录：{citation['quote']}")
                lines.append(f"   来源：{chunk['file']} 行 {chunk['start_line']}-{chunk['end_line']}")
            if 'notice' in entry:lines.append('   '+entry['notice'])
            elif title=='需要确认':lines.append('   请核实故事含义、适用设定或检索证据；当前不能作明确结论。')
    lines.extend(['','故事内部的事件检查：'])
    story=report['story_consistency']
    if story['status']=='not_run':lines.append('未执行；这份历史判断记录只包含与设定的核对。')
    else:
        lines.append(f"共{story['pair_count']}对事件，发现{len(story['contradictions'])}项冲突、{len(story['uncertainties'])}项需要确认。")
        for item in story['contradictions']+story['uncertainties']:
            lines.append(' - '+('段内矛盾' if item['verdict']=='contradiction' else '待确认')+'：'+item['reason'])
            for event in item['events']:lines.append('   事件：'+event['event'])
            for citation in item['citations']:lines.append('   故事原文：'+citation['quote'])
        if story.get('unprocessed_pair_count'):lines.append(f"还有{story['unprocessed_pair_count']}对未检查，不作无矛盾结论。")
    if report['processing_issues']:
        lines.extend(['','未完成的检查：'])
        for issue in report['processing_issues']:
            lines.append(' - '+issue.get('fact','上游处理未完整完成')+'：'+str(issue.get('detail',issue.get('details',''))))
    lines.extend(['','以上只覆盖已提取并核对的事实；“无矛盾条目”不证明整段故事无矛盾。'])
    return '\n'.join(lines)+'\n'


def validate_judge(report):
    if not isinstance(report,dict) or report.get('stage')!='judge' or report.get('status') not in ('ok','partial','error'):
        raise ValueError('report输入须为judge完整JSON报告')
    upstream=report.get('retrieval');rows=validate_retrieval(upstream)
    if report.get('events')!=upstream['events']:raise ValueError('判断事件与检索原事件不匹配')
    for key in ('pending_event_ids','ignored_event_ids'):
        if report.get(key)!=upstream[key]:raise ValueError(key+'与检索分区不匹配')
    if upstream['status']=='error' and report['status']!='error' or upstream['status']=='partial' and report['status']=='ok':
        raise ValueError('判断状态不能掩盖上游未完成')
    source={i['event_id']:i for i in upstream['items']};seen=set();registry={}
    items=report.get('items')
    if not isinstance(items,list):raise ValueError('判断items须为数组')
    for item in items:
        if not isinstance(item,dict) or not isinstance(item.get('event_id'),str) or item['event_id'] not in source or item['event_id'] in seen:
            raise ValueError('判断事件编号不存在或重复')
        eid=item['event_id'];seen.add(eid);original=source[eid]
        if item.get('event')!=original['event'] or item.get('status') not in ('ok','error','support_only'):raise ValueError('判断事件内容或状态无效')
        if (item['status']=='support_only')!=(original['status']=='support_only'):raise ValueError('判断支持事件路由不匹配')
        if item['status']=='support_only':
            if item.get('required_by_event_ids')!=rows[eid]['required_by_event_ids'] or 'verdict' in item:raise ValueError('支持事件不能独立判定')
            continue
        if item.get('evidence')!=original['evidence']:raise ValueError('判断证据与检索证据不匹配')
        if item.get('verdict') not in LABELS or not isinstance(item.get('reason'),str) or not item['reason'].strip():raise ValueError('判断类别或理由无效')
        if item['status']=='error' and item['verdict']!='uncertain':raise ValueError('失败事件不能给明确结论')
        if original['status']=='error' and item['status']!='error':raise ValueError('判断不能掩盖检索失败')
        citations=item.get('citations');aliases={'L'+str(i):c for i,c in enumerate(original['evidence'],1)};used=set()
        if not isinstance(citations,list):raise ValueError('引用须为数组')
        if item['verdict']!='uncertain' and not citations:raise ValueError('明确结论须有证据')
        for citation in citations:
            if not isinstance(citation,dict) or not isinstance(citation.get('evidence_id'),str):raise ValueError('引用无效')
            label=citation['evidence_id'];chunk=aliases.get(label)
            if chunk is None or label in used or citation.get('chunk_id')!=chunk['id']:raise ValueError('引用编号或canonical片段编号不匹配')
            quote=citation.get('quote')
            if not isinstance(quote,str) or not quote.strip() or quote not in chunk['text']:raise ValueError('引用原文不属于对应设定')
            used.add(label)
            record={k:copy.deepcopy(chunk[k]) for k in ('id','text','file','start_line','end_line','heading_path','title_path') if k in chunk}
            if chunk['id'] in registry and registry[chunk['id']]!=record:raise ValueError('同一片段编号的证据内容/来源冲突')
            registry[chunk['id']]=record
    if report['status']!='error' and seen!=set(source):raise ValueError('缺少判断记录')
    if report['status']=='ok' and (any(i['status']=='error' for i in items) or report['pending_event_ids']):raise ValueError('成功状态不能掩盖失败/待复核事件')
    if 'story_check' in report:
        from .story_check import validate_story_result
        filtering=report['retrieval']['filtering'];rows=validate_filtering(filtering)
        expected=[e['id'] for e in filtering['selected_events'] if e['modality']=='observed' and not rows[e['id']].get('support_only')]
        validate_story_result(report['story_check'],report['events'],expected)
        if report['story_check']['status']!='ok' and report['status']=='ok':raise ValueError('判断状态不能掩盖段内检查未完成')
    return registry


def build_report(judge_report):
    registry=validate_judge(judge_report)
    upstream=copy.deepcopy(judge_report);findings=[];groups={};failed=[];supports=[]
    counts={k:0 for k in ('consistent','contradiction','uncertain','failed','pending','ignored','support_only','unprocessed')}
    for item in upstream['items']:
        event=item['event'];eid=item['event_id']
        if item['status']=='support_only':
            supports.append(dict(event_id=eid,event=event,required_by_event_ids=item['required_by_event_ids']));counts['support_only']+=1;continue
        if item['status']=='error':
            failed.append(dict(event_id=eid,event=event,reason=item['reason'],error=item.get('error'),origin=item.get('origin')));counts['failed']+=1;continue
        verdict=item['verdict'];counts[verdict]+=1
        citations=[dict(chunk_id=c['chunk_id'],quote=c['quote']) for c in item['citations']]
        # Offsets and local context prevent merging repeated actions in different scenes.
        candidates=[{k:chunk[k] for k in ('id','text','file','start_line','end_line','heading_path','title_path') if k in chunk} for chunk in item['evidence']]
        identity=dict(event={k:event[k] for k in ('actors','event','mental_state','explicit','modality','conditions','source_ids','context_ids','sources','contexts') if k in event},verdict=verdict,
                      candidates=sorted(candidates,key=lambda c:c['id']),citations=sorted(citations,key=lambda c:(c['chunk_id'],c['quote'])),origin=item.get('origin'),assessment=item.get('assessment'),uncertainty_code=item.get('uncertainty_code'),reason_origin=item.get('reason_origin'))
        spans=event.get('sources',[])
        located=bool(event.get('source_ids')) or bool(spans) and all(type(s.get('start')) is int and type(s.get('end')) is int and 0<=s['start']<s['end'] for s in spans)
        if not located:identity['unlocated_event_id']=eid
        key=json.dumps(identity,ensure_ascii=False,sort_keys=True)
        if key not in groups:
            finding=dict(id='R'+str(len(findings)+1),event_ids=[],actors=event['actors'],event=event['event'],verdict=verdict,label=LABELS[verdict],reasons=[],citations=citations,
                         sources=event.get('sources',[]),contexts=event.get('contexts',[]),origin=item.get('origin'),assessment=item.get('assessment'),reason_origin=item.get('reason_origin'),uncertainty_code=item.get('uncertainty_code'))
            groups[key]=finding;findings.append(finding)
        finding=groups[key];finding['event_ids'].append(eid)
        if item['reason'] not in finding['reasons']:finding['reasons'].append(item['reason'])
    filtering=upstream['retrieval']['filtering'];seen={i['event_id'] for i in upstream['items']}
    unprocessed=[e for e in filtering['selected_events'] if e['id'] not in seen]
    counts.update(pending=len(upstream['pending_event_ids']),ignored=len(upstream['ignored_event_ids']),unprocessed=len(unprocessed))
    checked=sum(counts[k] for k in LABELS)
    story=upstream.get('story_check')
    story_conflicts=[i for i in story['items'] if i['status']=='ok' and i['verdict']=='contradiction'] if story else []
    story_uncertain=[i for i in story['items'] if i['status']=='ok' and i['verdict']=='uncertain'] if story else []
    verdict='contradiction' if counts['contradiction'] or story_conflicts else 'uncertain' if upstream['status']!='ok' or counts['uncertain'] or counts['pending'] or story_uncertain or not checked else 'consistent'
    labels={'contradiction':'已核对事实中发现明确矛盾','uncertain':'尚不能对已核对事实给出完整明确结论','consistent':'已核对事实明确吻合'}
    reviews=[dict(event_id=e['id'],reasons=e.get('review_reasons',[])) for e in upstream['events'] if e.get('review_required')]
    feedback=presentation_report(findings,counts,upstream['status'],failed,filtering['pending_events'],unprocessed,filtering,upstream.get('failure_reasons',{}))
    feedback['story_consistency']=dict(status=story['status'] if story else 'not_run',scope='current_story_observed_events',pair_count=story['pair_count'] if story else 0,unprocessed_pair_count=story['unprocessed_pair_count'] if story else 0,contradictions=story_conflicts,uncertainties=story_uncertain)
    if story:
        feedback['overview']+=f"段内事件检查另发现{len(story_conflicts)}项矛盾、{len(story_uncertain)}项待确认。"
        for item in story['items']:
            if item['status']=='error':feedback['processing_issues'].append(dict(kind='story_check_failure',fact=' / '.join(e['event'] for e in item['events']),detail=item['error']))
    return dict(schema_version='agent-pipeline-report-v2',stage='report',status=upstream['status'],report=feedback,
                summary=dict(verdict=verdict,label=labels[verdict],scope='checked_events',counts=counts,checked_event_count=checked,finding_count=len(findings)),
                findings=findings,evidence=list(registry.values()),failed_items=failed,pending_events=filtering['pending_events'],ignored_events=filtering['ignored_events'],
                support_events=supports,unprocessed_events=unprocessed,review_items=reviews,events=upstream['events'],judge=upstream,
                processing_complete=upstream['status']=='ok',review_required=bool(reviews) or verdict!='consistent' or upstream['status']!='ok',
                failure_reasons=upstream.get('failure_reasons',{}),semantic_verification='not_verified',notice=NOTICE)
