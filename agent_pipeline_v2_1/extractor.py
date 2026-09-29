"""Non-semantic wire repair around the frozen v2 extractor model adapter."""

from __future__ import annotations

import json


FIELD_ALIASES = {
    "checkreason": "check_reason",
    "checkReason": "check_reason",
    "contextips": "context_ids",
    "contextIds": "context_ids",
    "sourceIds": "source_ids",
    "mentalological": "mental_state",
    "mentalronic": "mental_state",
    "mentalState": "mental_state",
}
REASON_ALIASES = {
    "mechanibility": "mechanism",
    "mechanical": "mechanism",
    "observed": "state_time_space",
}
PLACEHOLDERS = {"未知人物", "未知主体", "某人"}
EXPLICIT_INDEFINITE_ACTORS = ("一个人", "一人", "有人", "某人")


class ExtractionRepairLLM:
    def __init__(self, delegate):
        self.delegate = delegate

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    @property
    def last_generation(self):
        return getattr(self.delegate, "last_generation", {})

    @last_generation.setter
    def last_generation(self, value):
        self.delegate.last_generation = value

    def _generate_batch(self, *args, **kwargs):
        return self.delegate._generate_batch(*args, **kwargs)

    def _generate(self, messages, max_new_tokens=1536):
        raw = self.delegate._generate(messages, max_new_tokens=max_new_tokens)
        try:
            value = json.loads(raw)
            payload = json.loads(messages[1]["content"])
        except (ValueError, TypeError, KeyError, IndexError):
            return raw
        if not isinstance(value, dict) or not isinstance(value.get("events"), list):
            return raw
        span_map = {}
        ordered_ids = []
        for group in (payload.get("context_spans", {}), payload.get("target_spans", {})):
            if isinstance(group, dict):
                span_map.update({key: text for key, text in group.items() if isinstance(text, str)})
                ordered_ids.extend(key for key, text in group.items() if isinstance(text, str))
        repaired_events = []
        for original in value["events"]:
            if not isinstance(original, dict):
                repaired_events.append(original)
                continue
            event = dict(original)
            for old, new in FIELD_ALIASES.items():
                if old in event and new not in event:
                    event[new] = event.pop(old)
            if event.get("check_reason") in REASON_ALIASES:
                event["check_reason"] = REASON_ALIASES[event["check_reason"]]
            actors = event.get("actors")
            source_ids = event.get("source_ids")
            context_ids = event.get("context_ids")
            if not isinstance(actors, list) or not isinstance(source_ids, list) or not isinstance(context_ids, list):
                repaired_events.append(event)
                continue
            referenced = "".join(span_map.get(span_id, "") for span_id in source_ids + context_ids)
            normalized_actors = []
            grounded = True
            for actor in actors:
                if not isinstance(actor, str):
                    grounded = False
                    break
                if actor in referenced:
                    normalized_actors.append(actor)
                    continue
                replacement = next((name for name in EXPLICIT_INDEFINITE_ACTORS if actor in PLACEHOLDERS and name in referenced), None)
                if replacement:
                    normalized_actors.append(replacement)
                    continue
                source_positions = [ordered_ids.index(span_id) for span_id in source_ids if span_id in ordered_ids]
                limit = min(source_positions) if source_positions else len(ordered_ids)
                matching_context = next((span_id for span_id in reversed(ordered_ids[:limit]) if actor in span_map[span_id] and span_id not in source_ids), None)
                if matching_context:
                    context_ids.append(matching_context)
                    referenced += span_map[matching_context]
                    normalized_actors.append(actor)
                    continue
                grounded = False
                break
            if grounded and normalized_actors:
                event["actors"] = normalized_actors
                event["context_ids"] = list(dict.fromkeys(context_ids))
                repaired_events.append(event)
        value["events"] = repaired_events
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
