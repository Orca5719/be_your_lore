"""Structured author entries, preview first and commit only after confirmation."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(text, encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class EntryStore:
    def __init__(self, path):
        self.path = Path(path)

    def _read(self):
        if not self.path.exists():
            return {'schema_version': 1, 'entries': []}, 'empty'
        raw = self.path.read_bytes()
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise ValueError('录入资料损坏，请恢复备份，不能覆盖') from exc
        if not isinstance(value, dict) or value.get('schema_version') != 1 or not isinstance(value.get('entries'), list):
            raise ValueError('录入资料格式不匹配')
        required = {'id','entity','category','text','event_time','created_at'}
        seen = set()
        for entry in value['entries']:
            if not isinstance(entry, dict) or set(entry) != required:
                raise ValueError('录入条目损坏')
            self._validate(entry['entity'], entry['category'], entry['text'], entry['event_time'])
            if not isinstance(entry['id'], str) or entry['id'] in seen or not isinstance(entry['created_at'], str):
                raise ValueError('录入条目 ID 或时间无效')
            seen.add(entry['id'])
        return value, hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _validate(entity, category, text, event_time):
        for name, value in [('实体', entity), ('类别', category), ('内容', text)]:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(name + '不能为空')
        if any('\n' in x or '\r' in x for x in [entity, category]):
            raise ValueError('实体与类别必须是单行文字')
        if event_time is not None and (not isinstance(event_time, str) or not event_time.strip() or '\n' in event_time or '\r' in event_time):
            raise ValueError('事件时间必须是非空单行文字或未指定')

    def preview(self, entity, category, text, chunks, event_time=None):
        self._validate(entity, category, text, event_time)
        entity, category, text = entity.strip(), category.strip(), text.strip()
        value, revision = self._read()
        grouped = {}
        for chunk in chunks:
            headings = chunk.get('heading_path', [])
            if len(headings) >= 2 and headings[-2] == entity:
                grouped.setdefault(headings[-1], []).append(dict(text=chunk['text'], source=dict(file=chunk['file'], start_line=chunk['start_line'], end_line=chunk['end_line']), id=chunk['id']))
        for entry in value['entries']:
            if entry['entity'] == entity:
                grouped.setdefault(entry['category'], []).append(dict(entry, source={'file':str(self.path)}))
        # Generated Markdown is omitted by the caller, so authored entries appear once.
        return dict(status='pending_confirmation', store_revision=revision, entity=entity, category=category,
            proposed=dict(entity=entity, category=category, text=text, event_time=event_time),
            existing=grouped.pop(category, []), other_categories=grouped)

    def commit(self, preview):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix(self.path.suffix + '.lock')
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ValueError('另一个录入正在保存；若上次异常退出，请检查 lock 文件') from exc
        try:
            os.close(descriptor)
            value, revision = self._read()
            if preview.get('store_revision') != revision:
                raise ValueError('预览后资料已改变，请重新预览再确认')
            proposed = preview['proposed']
            self._validate(proposed['entity'], proposed['category'], proposed['text'], proposed['event_time'])
            if any(all(entry[key] == proposed[key] for key in ('entity','category','text','event_time')) for entry in value['entries']):
                raise ValueError('相同条目已存在，未重复保存')
            entry = dict(proposed, id=uuid.uuid4().hex, created_at=datetime.now(timezone.utc).isoformat())
            value['entries'].append(entry)
            atomic_write(self.path, json.dumps(value, ensure_ascii=False, indent=2))
            return entry
        finally:
            lock.unlink(missing_ok=True)

    def export(self, path):
        path = Path(path)
        if path.exists() and not path.read_text(encoding='utf-8').startswith('# 用户确认录入（由 entries.json 生成，请勿直接编辑）'):
            raise ValueError('导出目标已有手写资料，拒绝覆盖：' + str(path))
        value, _ = self._read()
        lines = ['# 用户确认录入（由 entries.json 生成，请勿直接编辑）', '']
        for entry in value['entries']:
            lines += ['## ' + entry['entity'], '### ' + entry['category'], '']
            if entry['event_time']:
                lines += ['    事件时间：' + entry['event_time']]
            lines += ['    ' + line for line in entry['text'].splitlines()]
            lines += ['']
        atomic_write(path, '\n'.join(lines))

