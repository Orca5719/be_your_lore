"""Validate citations and structure; this does not verify semantic reasoning."""
import json


def validate_answer(raw, text, evidence):
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError('模型输出不是有效 JSON') from exc
    if not isinstance(value, dict) or set(value) != {'findings'}:
        raise ValueError('输出必须包含 findings 数组')
    findings = value['findings']
    if not isinstance(findings, list) or not findings:
        raise ValueError('findings 不能为空')
    for finding in findings:
        required = {'input_quote', 'verdict', 'evidence', 'reason', 'new_candidates'}
        if not isinstance(finding, dict) or set(finding) != required:
            raise ValueError('判断字段不符合协议')
        quote = finding['input_quote']
        if not isinstance(quote, str) or not quote.strip() or quote not in text:
            raise ValueError('input_quote 必须来自原输入')
        if finding['verdict'] not in ('一致', '矛盾', '不确定'):
            raise ValueError('无效判断类别')
        if not isinstance(finding['reason'], str) or not finding['reason'].strip():
            raise ValueError('原因不能为空')
        candidates = finding['new_candidates']
        if not isinstance(candidates, list) or any(not isinstance(x, str) or not x.strip() for x in candidates):
            raise ValueError('新增候选必须是非空字符串数组')
        if candidates and (finding['verdict'] != '不确定' or candidates != [quote]):
            raise ValueError('新增候选只能是不确定条目的 input_quote 原文，不能扩展或重复')
        citations = finding['evidence']
        if not isinstance(citations, list):
            raise ValueError('evidence 必须是数组')
        if finding['verdict'] in ('一致', '矛盾') and not citations:
            raise ValueError('一致或矛盾必须提供证据')
        for citation in citations:
            if not isinstance(citation, dict) or set(citation) != {'chunk_id', 'quote'}:
                raise ValueError('引用字段不符合协议')
            chunk_id, cited = citation['chunk_id'], citation['quote']
            if not isinstance(chunk_id, str) or chunk_id not in evidence:
                raise ValueError('引用了不存在的片段')
            if not isinstance(cited, str) or not cited.strip() or cited not in evidence[chunk_id]:
                raise ValueError('引用原文不在证据中')
    verdicts = {x['verdict'] for x in findings}
    overall = '矛盾' if '矛盾' in verdicts else ('不确定' if '不确定' in verdicts else '一致')
    # Review candidates are uncertain original claims, not confirmed novel facts.
    review_candidates = list(dict.fromkeys(x['input_quote'] for x in findings if x['verdict'] == '不确定'))
    return dict(status='ok', overall_verdict=overall, findings=findings, review_candidates=review_candidates)




def validate_retrieval_answer(raw, text, results):
    """Bind short, session-local evidence labels to canonical chunk IDs."""
    aliases = {'E'+str(i): row['id'] for i,row in enumerate(results,1)}
    evidence = {row['id']:row['text'] for row in results}
    try:
        value = json.loads(raw)
    except (ValueError,TypeError) as exc:
        raise ValueError('模型输出不是有效 JSON') from exc
    if isinstance(value,dict) and isinstance(value.get('findings'),list):
        for finding in value['findings']:
            if isinstance(finding,dict) and isinstance(finding.get('evidence'),list):
                for citation in finding['evidence']:
                    if isinstance(citation,dict):
                        label=citation.get('chunk_id')
                        if not isinstance(label,str) or label not in aliases:
                            raise ValueError('模型引用了不存在的证据编号')
                        citation['chunk_id']=aliases[label]
    return validate_answer(json.dumps(value,ensure_ascii=False),text,evidence)
