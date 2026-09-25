"""Schema validation and legacy adaptation for the expanded story benchmark."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


VERDICTS = ("一致", "矛盾", "不确定")
GROUP_CONFLICT_COUNTS = {"zero_conflict": 0, "one_conflict": 1, "multi_conflict": 2}
DIMENSIONS = {
    "time", "character_knowledge", "space", "character_relation",
    "physical_rule", "world_rule", "causality", "identity",
}
IGNORED_REASONS = {"routine", "process_detail"}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label}必须是对象")
    return value


def _require_list(value: Any, label: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{label}必须是列表")
    return value


def _required(row: dict, fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"{label}缺少字段：{', '.join(missing)}")


def _unique_ids(rows: list, label: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        _require_mapping(row, label)
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{label}的ID不能为空")
        if identifier in result:
            raise ValueError(f"{label}ID重复：{identifier}")
        result[identifier] = row
    return result


def _validate_anchor(anchor: Any, story: str, label: str) -> tuple[int, int]:
    row = _require_mapping(anchor, label)
    _required(row, ("text", "start", "end"), label)
    text, start, end = row["text"], row["start"], row["end"]
    if (
        not isinstance(text, str)
        or not text
        or type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(story)
        or story[start:end] != text
    ):
        raise ValueError(f"{label}不一致")
    return start, end


def _validate_evidence_sets(
    evidence_sets: Any,
    relevant_ids: set[str],
    label: str,
    *,
    required: bool,
) -> None:
    rows = _require_list(evidence_sets, label)
    if required and not rows:
        raise ValueError(f"{label}不能为空")
    for evidence_set in rows:
        if (
            not isinstance(evidence_set, list)
            or not evidence_set
            or len(evidence_set) != len(set(evidence_set))
            or any(not isinstance(chunk_id, str) or chunk_id not in relevant_ids for chunk_id in evidence_set)
        ):
            raise ValueError(f"{label}必须是relevant_lore_ids的非空子集")


def validate_story_dataset(
    data: dict,
    root: Path,
    index_dir: Path,
    *,
    expected_cases: int = 24,
    expected_facts_per_verdict: int | None = 24,
) -> dict:
    """Validate expanded annotations against their story, lore corpus, and index."""
    from index_store import load_index

    dataset = _require_mapping(data, "故事测试集")
    _required(dataset, ("schema_version", "name", "version", "status", "notice", "corpus_snapshot", "cases"), "故事测试集")
    if dataset["schema_version"] != "agent-pipeline-v2-story-dataset-v2":
        raise ValueError("故事测试集schema_version不受支持")
    cases = _require_list(dataset["cases"], "案例")
    if len(cases) != expected_cases:
        raise ValueError(f"案例数量应为{expected_cases}，实际为{len(cases)}")
    case_map = _unique_ids(cases, "案例")

    root = Path(root)
    index_dir = Path(index_dir)
    snapshot = _require_mapping(dataset["corpus_snapshot"], "资料快照")
    _required(snapshot, ("files_sha256", "index_version", "chunk_count"), "资料快照")
    lore_root = root / "lore"
    actual_files = {
        path.relative_to(lore_root).as_posix(): _digest(path)
        for path in lore_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    }
    if snapshot["files_sha256"] != actual_files:
        raise ValueError("资料摘要不匹配，拒绝混用不同世界观基线")
    current_version = (index_dir / "CURRENT").read_text(encoding="utf-8").strip()
    if snapshot["index_version"] != current_version:
        raise ValueError("索引版本与测试集快照不一致")
    _, metadata = load_index(index_dir)
    chunks = _unique_ids(_require_list(metadata.get("chunks"), "索引片段"), "索引片段")
    if snapshot["chunk_count"] != len(chunks):
        raise ValueError("索引片段数不匹配")

    verdicts: Counter[str] = Counter()
    groups: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    global_ids: set[str] = set()
    finding_count = 0
    ignored_count = 0
    non_event_count = 0

    for case_id, case in case_map.items():
        _required(case, (
            "id", "title", "group", "dimensions", "story", "character_count",
            "gold_facts", "ignored_events", "non_events", "gold_findings", "annotation_status",
        ), f"案例{case_id}")
        story = case["story"]
        if not isinstance(story, str) or not 200 <= len(story) <= 800:
            raise ValueError(f"故事必须为200–800字符：{case_id}")
        if case["character_count"] != len(story):
            raise ValueError(f"故事字符数记录错误：{case_id}")
        group = case["group"]
        if group not in GROUP_CONFLICT_COUNTS:
            raise ValueError(f"案例分组无效：{case_id}")
        groups[group] += 1
        case_dimensions = _require_list(case["dimensions"], f"案例维度：{case_id}")
        if any(dimension not in DIMENSIONS for dimension in case_dimensions):
            raise ValueError(f"案例包含未知维度：{case_id}")

        facts = _require_list(case["gold_facts"], f"金标事实：{case_id}")
        fact_map = _unique_ids(facts, "事实")
        for fact_id in fact_map:
            global_id = f"{case_id}/{fact_id}"
            if global_id in global_ids:
                raise ValueError(f"全局事实ID重复：{global_id}")
            global_ids.add(global_id)
        conflict_count = 0
        occupied_anchors: set[tuple[int, int]] = set()
        for fact_id, fact in fact_map.items():
            label = f"事实{case_id}/{fact_id}"
            _required(fact, (
                "id", "subject", "predicate", "object", "normalized_fact", "type", "dimension",
                "source_anchors", "context_anchors", "expected_verdict", "relevant_lore_ids",
                "minimum_evidence_sets", "reportable",
            ), label)
            for text_field in ("subject", "predicate", "normalized_fact", "type"):
                if not isinstance(fact[text_field], str) or not fact[text_field].strip():
                    raise ValueError(f"{label}的{text_field}不能为空")
            if fact["object"] is not None and not isinstance(fact["object"], str):
                raise ValueError(f"{label}的object必须是文字或null")
            if fact["dimension"] not in DIMENSIONS:
                raise ValueError(f"{label}的dimension无效")
            dimensions[fact["dimension"]] += 1
            verdict = fact["expected_verdict"]
            if verdict not in VERDICTS:
                raise ValueError(f"{label}的金标判断无效")
            verdicts[verdict] += 1
            conflict_count += verdict == "矛盾"
            if type(fact["reportable"]) is not bool:
                raise ValueError(f"{label}的reportable必须为布尔值")
            source_anchors = _require_list(fact["source_anchors"], f"{label}金标原文位置")
            if not source_anchors:
                raise ValueError(f"{label}金标原文位置不能为空")
            for anchor in source_anchors:
                occupied_anchors.add(_validate_anchor(anchor, story, f"{label}金标原文位置"))
            for anchor in _require_list(fact["context_anchors"], f"{label}上下文位置"):
                _validate_anchor(anchor, story, f"{label}上下文位置")

            relevant = _require_list(fact["relevant_lore_ids"], f"{label}相关设定片段")
            if len(relevant) != len(set(relevant)) or any(chunk_id not in chunks for chunk_id in relevant):
                raise ValueError(f"{label}引用了不存在或重复的设定片段")
            _validate_evidence_sets(
                fact["minimum_evidence_sets"], set(relevant), f"{label}最小证据组合",
                required=verdict != "不确定",
            )
        if conflict_count != GROUP_CONFLICT_COUNTS[group]:
            raise ValueError(f"案例分组与矛盾数量不一致：{case_id}")

        ignored = _require_list(case["ignored_events"], f"忽略事件：{case_id}")
        ignored_count += len(ignored)
        for number, item in enumerate(ignored, 1):
            item = _require_mapping(item, f"忽略事件{case_id}/{number}")
            _required(item, ("anchor", "normalized_proposition", "reason"), f"忽略事件{case_id}/{number}")
            anchor_key = _validate_anchor(item["anchor"], story, f"忽略事件位置：{case_id}")
            if anchor_key in occupied_anchors:
                raise ValueError(f"同一原文不能同时标为金标事实和忽略事件：{case_id}")
            if not isinstance(item["normalized_proposition"], str) or not item["normalized_proposition"].strip():
                raise ValueError(f"忽略事件缺少规范命题：{case_id}")
            if item["reason"] not in IGNORED_REASONS:
                raise ValueError(f"忽略事件原因无效：{case_id}")

        non_events = _require_list(case["non_events"], f"非事件：{case_id}")
        non_event_count += len(non_events)
        for number, item in enumerate(non_events, 1):
            item = _require_mapping(item, f"非事件{case_id}/{number}")
            _required(item, ("anchor", "reason"), f"非事件{case_id}/{number}")
            _validate_anchor(item["anchor"], story, f"非事件位置：{case_id}")
            if not isinstance(item["reason"], str) or not item["reason"].strip():
                raise ValueError(f"非事件原因不能为空：{case_id}")

        findings = _require_list(case["gold_findings"], f"金标结论：{case_id}")
        finding_map = _unique_ids(findings, "结论")
        finding_count += len(finding_map)
        for finding_id, finding in finding_map.items():
            label = f"结论{case_id}/{finding_id}"
            _required(finding, ("id", "gold_fact_ids", "expected_type", "dimension", "minimum_evidence_sets", "merge"), label)
            fact_ids = _require_list(finding["gold_fact_ids"], f"{label}事实引用")
            if not fact_ids or len(fact_ids) != len(set(fact_ids)) or any(fact_id not in fact_map for fact_id in fact_ids):
                raise ValueError(f"{label}引用了不存在的事实")
            if finding["expected_type"] != "conflict":
                raise ValueError(f"{label}的expected_type必须为conflict")
            if finding["dimension"] not in DIMENSIONS:
                raise ValueError(f"{label}的dimension无效")
            if any(fact_map[fact_id]["expected_verdict"] != "矛盾" or not fact_map[fact_id]["reportable"] for fact_id in fact_ids):
                raise ValueError(f"{label}只能引用可报告的矛盾事实")
            relevant_union = {chunk_id for fact_id in fact_ids for chunk_id in fact_map[fact_id]["relevant_lore_ids"]}
            _validate_evidence_sets(finding["minimum_evidence_sets"], relevant_union, f"{label}最小证据组合", required=True)
            if type(finding["merge"]) is not bool:
                raise ValueError(f"{label}的merge必须为布尔值")

        reportable_conflicts = {fact_id for fact_id, fact in fact_map.items() if fact["expected_verdict"] == "矛盾" and fact["reportable"]}
        covered_conflicts = {fact_id for finding in findings for fact_id in finding["gold_fact_ids"]}
        if reportable_conflicts != covered_conflicts:
            raise ValueError(f"可报告矛盾与金标结论覆盖不一致：{case_id}")

    if expected_facts_per_verdict is not None:
        expected = {verdict: expected_facts_per_verdict for verdict in VERDICTS}
        actual = {verdict: verdicts[verdict] for verdict in VERDICTS}
        if actual != expected:
            raise ValueError(f"判断分布不匹配：应为{expected}，实际为{actual}")
    if expected_cases == 24:
        expected_groups = {group: 8 for group in GROUP_CONFLICT_COUNTS}
        actual_groups = {group: groups[group] for group in GROUP_CONFLICT_COUNTS}
        if actual_groups != expected_groups:
            raise ValueError(f"案例分组分布不匹配：应为{expected_groups}，实际为{actual_groups}")

    return {
        "cases": len(cases),
        "gold_facts": sum(verdicts.values()),
        "gold_findings": finding_count,
        "ignored_events": ignored_count,
        "non_events": non_event_count,
        "index_chunks": len(chunks),
        "dataset_status": dataset["status"],
        "groups": dict(groups),
        "verdicts": {verdict: verdicts[verdict] for verdict in VERDICTS},
        "dimensions": dict(dimensions),
        "snapshot": {"index_version": current_version, "files_sha256": actual_files},
    }


def load_story_dataset(
    path: Path,
    root: Path,
    index_dir: Path,
    *,
    expected_cases: int = 24,
    expected_facts_per_verdict: int | None = 24,
) -> tuple[dict, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    validation = validate_story_dataset(
        data, root, index_dir,
        expected_cases=expected_cases,
        expected_facts_per_verdict=expected_facts_per_verdict,
    )
    return data, validation


def upgrade_legacy_story_dataset(data: dict) -> dict:
    """Convert the frozen four-story pilot in memory for Task 1 validation only."""
    if data.get("schema_version") == "agent-pipeline-v2-story-dataset-v2":
        return data
    converted = {
        "schema_version": "agent-pipeline-v2-story-dataset-v2",
        "name": data.get("name", "Story-Level Consistency Benchmark"),
        "version": data.get("version", "legacy"),
        "status": data.get("status", "legacy"),
        "notice": data.get("notice", ""),
        "corpus_snapshot": data["corpus_snapshot"],
        "cases": [],
    }
    for old_case in data["cases"]:
        facts = []
        findings = []
        case_dimensions: list[str] = []
        for old_fact in old_case["gold_facts"]:
            dimension = "physical_rule"
            relevant_ids = [row["id"] for row in old_fact.get("canonical_evidence", [])]
            reportable = old_fact["expected_verdict"] == "矛盾"
            fact = {
                "id": old_fact["id"],
                "subject": old_fact["subject"],
                "predicate": old_fact["predicate"],
                "object": old_fact.get("object"),
                "normalized_fact": "".join(str(old_fact.get(key) or "") for key in ("subject", "predicate", "object")),
                "type": old_fact["type"],
                "dimension": dimension,
                "source_anchors": [old_fact["source_anchor"]],
                "context_anchors": old_fact.get("context_anchors", []),
                "expected_verdict": old_fact["expected_verdict"],
                "relevant_lore_ids": relevant_ids,
                "minimum_evidence_sets": old_fact.get("acceptable_evidence_sets", []),
                "reportable": reportable,
            }
            facts.append(fact)
            if dimension not in case_dimensions:
                case_dimensions.append(dimension)
            if reportable:
                findings.append({
                    "id": "F" + old_fact["id"].lstrip("G"),
                    "gold_fact_ids": [old_fact["id"]],
                    "expected_type": "conflict",
                    "dimension": dimension,
                    "minimum_evidence_sets": old_fact["acceptable_evidence_sets"],
                    "merge": False,
                })
        converted["cases"].append({
            "id": old_case["id"],
            "title": old_case["title"],
            "group": old_case["group"],
            "dimensions": case_dimensions,
            "story": old_case["story"],
            "character_count": old_case["character_count"],
            "gold_facts": facts,
            "ignored_events": [
                {"anchor": item["anchor"], "normalized_proposition": item["anchor"]["text"], "reason": "routine"}
                for item in old_case.get("ignored_examples", [])
            ],
            "non_events": [],
            "gold_findings": findings,
            "annotation_status": old_case["annotation_status"],
        })
    return converted
