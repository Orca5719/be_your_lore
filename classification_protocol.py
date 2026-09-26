"""Validate automatic labels; model suggestions never imply author approval."""
import json

def validate_classification(raw, text, categories, overrides=None):
    try:
        value = json.loads(raw)
    except (ValueError,TypeError) as exc:
        raise ValueError('自动分类输出不是有效 JSON，请手动指定字段') from exc
    if not isinstance(value,dict) or set(value) != {'entity','category','event_time'}:
        raise ValueError('自动分类字段无效')
    overrides = overrides or {}
    resolved = {}
    for name in ('entity','category','event_time'):
        candidate = overrides.get(name) if overrides.get(name) is not None else value[name]
        if candidate is not None and (not isinstance(candidate,str) or not candidate.strip() or '\n' in candidate or '\r' in candidate):
            raise ValueError('自动分类字段必须是单行文字或 null：'+name)
        resolved[name] = candidate.strip() if candidate else None
    if resolved['entity'] and overrides.get('entity') is None and resolved['entity'] not in text:
        raise ValueError('自动识别的实体不在原文中，请手动指定')
    if resolved['category'] and overrides.get('category') is None and resolved['category'] not in categories:
        raise ValueError('自动识别的类别不在可用分类中')
    if resolved['event_time'] and overrides.get('event_time') is None and resolved['event_time'] not in text:
        raise ValueError('自动识别的事件时间不在原文中')
    if overrides.get('entity') is None and resolved['entity'] in {'他','她','它','他们','她们','它们','这个人','此人','我','你','我们','你们'}:
        resolved['entity'] = None
    missing = [name for name in ('entity','category') if resolved[name] is None]
    return dict(resolved,status='needs_clarification' if missing else 'ok',missing=missing)

