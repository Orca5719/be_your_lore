"""Generation boundary adapter; preserves both raw wire and expanded output."""
import copy
import hashlib
import json
from pathlib import Path
from .wire import expand_payload

ROOT = Path(__file__).resolve().parents[1]


class WireBatchModel:
    def __init__(self, delegate, protocol, guard=None, repair_policy="none", prompt_policy="original"):
        if protocol not in ('original', 'compact'):
            raise ValueError('invalid wire protocol')
        if repair_policy not in ('none','schema_aliases_v1') or (protocol!='original' and repair_policy!='none'):
            raise ValueError('5C1 only varies schema repair on the original protocol')
        if prompt_policy not in ('original','semantic_v1') or (prompt_policy!='original' and (protocol!='original' or repair_policy!='none')):
            raise ValueError('5D0 varies only the original-protocol prompt, without schema repair')
        self.prompt_policy=prompt_policy
        self.repair_policy=repair_policy
        self.delegate, self.protocol, self.guard = delegate, protocol, guard
        self.original_prompt = (ROOT / 'agent_pipeline_v2/prompts/extractor_v1.txt').read_text(encoding='utf-8')
        self.compact_prompt = (Path(__file__).parent / 'prompts/extractor_compact.txt').read_text(encoding='utf-8')
        self.system_prompt=self.compact_prompt if protocol=='compact' else self.original_prompt
        if prompt_policy=='semantic_v1':
            self.system_prompt=(Path(__file__).parent/'prompts/extractor_semantic_v1.txt').read_text(encoding='utf-8')
        self.system_prompt_sha256=hashlib.sha256(self.system_prompt.encode('utf-8')).hexdigest()
        self.wire_records = []
        self.fatal_error = None

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    @property
    def last_batch_generation(self):
        return self.delegate.last_batch_generation

    @last_batch_generation.setter
    def last_batch_generation(self, value):
        self.delegate.last_batch_generation = value

    def _generate_batch(self, message_batches, max_new_tokens=1536):
        if self.fatal_error:
            raise RuntimeError(self.fatal_error)
        if len(message_batches) != 1:
            raise ValueError('5B uses batch=1 for every arm')
        if self.guard:
            try:
                self.guard()
            except RuntimeError as exc:
                self.fatal_error = str(exc)
                raise
        messages = copy.deepcopy(message_batches)
        if messages[0][0] != {'role': 'system', 'content': self.original_prompt}:
            raise ValueError('unexpected extractor system prompt')
        messages[0][0]['content'] = self.system_prompt
        try:
            outputs = self.delegate._generate_batch(messages, max_new_tokens=max_new_tokens)
        except Exception as exc:
            if 'out of memory' in str(exc).lower():
                self.fatal_error = str(exc)
            self.wire_records.append({'call_id': self.delegate.last_batch_generation.get('call_id'),
                                      'raw_output': getattr(exc, 'raw_output', None),
                                      'expanded_output': None, 'error': str(exc),
                                      'prompt_policy':self.prompt_policy,'system_prompt_sha256':self.system_prompt_sha256})
            raise
        if len(outputs) != 1:
            raise RuntimeError('unexpected generation output count')
        raw = outputs[0]
        error = None
        expanded = raw
        schema_repairs = []
        if self.repair_policy=='schema_aliases_v1':
            from .repair import repair_raw
            expanded,schema_repairs=repair_raw(raw)
        if self.protocol == 'compact':
            try:
                expanded = json.dumps(expand_payload(json.loads(raw)), ensure_ascii=False, separators=(',', ':'))
            except (ValueError, TypeError) as exc:
                error = str(exc)
                # Frozen window validation must fail/retry, never silently drop events.
                expanded = json.dumps({'compact_wire_error': error}, ensure_ascii=False)
        self.wire_records.append({'call_id': self.delegate.last_batch_generation.get('call_id'),
                                  'raw_output': raw, 'expanded_output': expanded, 'error': error,
                                  'schema_repairs':schema_repairs,
                                  'prompt_policy':self.prompt_policy,'system_prompt_sha256':self.system_prompt_sha256})
        return [expanded]
