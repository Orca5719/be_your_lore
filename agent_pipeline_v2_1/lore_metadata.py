"""Deterministic, non-LLM metadata generation for lore chunks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from collections import Counter


SCHEMA_VERSION = "agent-pipeline-v2.1-lore-metadata-v1"
GENERATOR_VERSION = "rules-v1"
TIME_RE = re.compile(r"(?:[0-9]{3,4}年(?:[0-9]{1,2}月(?:[0-9]{1,2}日)?)?|第[零〇一二三四五六七八九十百千万0-9]+[话章节卷幕])")

DIMENSION_RULES = {
    "physical_rule": ("生理", "心脏", "能力", "力量", "治疗", "机械", "义肢", "装备", "装甲", "续航", "冷却", "感知", "身体"),
    "character_knowledge": ("知道", "不知道", "认知", "保密", "告知", "秘密"),
    "character_relation": ("关系", "见面", "交谈", "父亲", "母亲", "契约", "宿主"),
    "identity": ("身份", "伪装", "代号", "真实姓名"),
    "space": ("地点", "位置", "城区", "地下", "广场", "教堂", "避难所", "档案馆", "交通", "列车"),
    "causality": ("导致", "因此", "使得", "使其", "后会", "才会", "触发", "副作用", "代价"),
    "world_rule": ("规则", "必须", "无法", "不能", "最多", "只有", "限制", "条件"),
}


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _domain(chunk: dict) -> str:
    root = (chunk.get("heading_path") or [""])[0]
    filename = Path(chunk.get("file", "")).stem.lower()
    candidates = root + " " + filename
    for domain, words in {
        "character": ("人物", "character"),
        "angel": ("天使", "angel"),
        "history": ("历史", "事件", "history"),
        "technology": ("科技", "技术", "technology"),
    }.items():
        if any(word in candidates for word in words):
            return domain
    return "other"


def _scope(chunk: dict) -> str | None:
    headings = chunk.get("heading_path") or []
    if len(headings) < 2 or TIME_RE.fullmatch(headings[-2]):
        return None
    return headings[-2].strip() or None


def _dimension_tags(chunk: dict, times: list[str]) -> list[str]:
    headings = chunk.get("heading_path") or []
    haystack = " ".join([*headings, chunk.get("text", "")])
    tags = {name for name, words in DIMENSION_RULES.items() if any(word in haystack for word in words)}
    if times or _domain(chunk) == "history":
        tags.add("time")
    return sorted(tags)


def _polarity(text: str) -> list[str]:
    values = []
    if any(word in text for word in ("不", "无", "未", "没")):
        values.append("negative")
    if any(word in text for word in ("必须", "只有", "最多", "不能", "无法", "限制")):
        values.append("restrictive")
    return values or ["affirmative"]


def build_lore_metadata(chunks: list[dict], index_version: str) -> dict:
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("lore chunks不能为空")
    if not isinstance(index_version, str) or not index_version.strip():
        raise ValueError("index_version不能为空")
    ordered = sorted(chunks, key=lambda row: row["id"])
    if len({row.get("id") for row in ordered}) != len(ordered):
        raise ValueError("chunk id缺失或重复")
    vocabulary = sorted({scope for chunk in ordered if (scope := _scope(chunk))}, key=lambda value: (-len(value), value))
    items = []
    for chunk in ordered:
        headings = chunk.get("heading_path") or []
        text = chunk.get("text", "")
        primary_scope = _scope(chunk)
        entities = sorted({term for term in vocabulary if term in text or term == primary_scope})
        times = sorted(set(TIME_RE.findall(" ".join([*headings, text]))))
        items.append({
            "chunk_id": chunk["id"],
            "source": {
                "file": chunk.get("file"),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
                "heading_path": headings,
            },
            "domain": _domain(chunk),
            "primary_scope": primary_scope,
            "category": headings[-1] if headings else None,
            "entities": entities,
            "time_markers": times,
            "locations": [],
            "dimension_tags": _dimension_tags(chunk, times),
            "polarity_tags": _polarity(text),
            "provenance": {
                "domain": "root_heading_or_file_rule",
                "primary_scope": "heading_path_penultimate_non_time",
                "category": "heading_path_last",
                "entities": "heading_vocabulary_exact_match",
                "time_markers": "explicit_regex_match",
                "locations": "not_inferred",
                "dimension_tags": "keyword_multilabel_rules",
                "polarity_tags": "surface_marker_rules",
            },
        })
    digest = hashlib.sha256(_canonical(items)).hexdigest()
    domain_counts = Counter(row["domain"] for row in items)
    dimension_counts = Counter(tag for row in items for tag in row["dimension_tags"])
    coverage = {}
    for field in ("primary_scope", "entities", "time_markers", "dimension_tags"):
        count = sum(bool(row[field]) for row in items)
        coverage[field] = {"count": count, "rate": count / len(items)}
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": {"name": "deterministic_lore_metadata", "version": GENERATOR_VERSION, "uses_llm": False},
        "index_version": index_version,
        "chunk_count": len(items),
        "entity_vocabulary": sorted(vocabulary),
        "metadata_digest": digest,
        "summary": {
            "domain_counts": dict(sorted(domain_counts.items())),
            "dimension_counts": dict(sorted(dimension_counts.items())),
            "coverage": coverage,
        },
        "items": items,
    }


def validate_lore_metadata(value: dict, chunks: list[dict], expected_index_version: str) -> None:
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("lore metadata schema_version无效")
    if value.get("index_version") != expected_index_version:
        raise ValueError("lore metadata与索引版本不匹配，必须重建")
    items = value.get("items")
    if not isinstance(items, list) or value.get("chunk_count") != len(items):
        raise ValueError("lore metadata数量无效")
    expected_ids = sorted(chunk["id"] for chunk in chunks)
    actual_ids = [row.get("chunk_id") for row in items]
    if actual_ids != expected_ids:
        raise ValueError("lore metadata与索引片段不匹配，必须重建")
    if value.get("metadata_digest") != hashlib.sha256(_canonical(items)).hexdigest():
        raise ValueError("lore metadata摘要无效")
    if value.get("generator", {}).get("uses_llm") is not False:
        raise ValueError("lore metadata生成器声明无效")
