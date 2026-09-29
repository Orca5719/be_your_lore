"""Judge 2.1 answer contract and deterministic coexistence guard."""

from __future__ import annotations

import copy


LABELS = {"consistent", "contradiction", "uncertain"}
RELATIONS = {"direct_support", "direct_conflict", "insufficient"}
ASSESSMENT_FIELDS = {
    "same_subject",
    "evidence_applicable",
    "relation",
    "can_both_be_true",
    "assumptions",
}


def _contradiction_is_proven(assessment: dict) -> bool:
    return (
        assessment["same_subject"] is True
        and assessment["evidence_applicable"] is True
        and assessment["relation"] == "direct_conflict"
        and assessment["can_both_be_true"] is False
        and not assessment["assumptions"]
    )


def _support_is_proven(assessment: dict) -> bool:
    return (
        assessment["same_subject"] is True
        and assessment["evidence_applicable"] is True
        and assessment["relation"] == "direct_support"
        and assessment["can_both_be_true"] is True
        and not assessment["assumptions"]
    )


def validate_model_answer(value: dict, evidence: list[dict]) -> None:
    if not isinstance(value, dict) or set(value) != {"verdict", "citations", "reason", "assessment"}:
        raise ValueError("判断字段必须为verdict/citations/reason/assessment")
    verdict = value["verdict"]
    if verdict not in LABELS:
        raise ValueError("verdict无效")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("reason不能为空")
    assessment = value["assessment"]
    if not isinstance(assessment, dict) or set(assessment) != ASSESSMENT_FIELDS:
        raise ValueError("assessment字段无效")
    for field in ("same_subject", "evidence_applicable", "can_both_be_true"):
        if assessment[field] is not None and type(assessment[field]) is not bool:
            raise ValueError(f"assessment.{field}必须为布尔值或null")
    if assessment["relation"] not in RELATIONS:
        raise ValueError("assessment.relation无效")
    assumptions = assessment["assumptions"]
    if not isinstance(assumptions, list) or any(not isinstance(item, str) or not item.strip() for item in assumptions):
        raise ValueError("assumptions须为文字数组")
    citations = value["citations"]
    if not isinstance(citations, list) or len(citations) > len(evidence):
        raise ValueError("citations无效")
    if verdict != "uncertain" and not citations:
        raise ValueError("明确结论必须引用证据")
    aliases = {f"L{number}": chunk for number, chunk in enumerate(evidence, 1)}
    seen = set()
    for citation in citations:
        if not isinstance(citation, dict) or set(citation) != {"evidence_id", "quote"}:
            raise ValueError("citation字段无效")
        alias = citation["evidence_id"]
        quote = citation["quote"]
        if alias not in aliases or alias in seen:
            raise ValueError("citation编号不存在或重复")
        if not isinstance(quote, str) or not quote.strip() or quote not in aliases[alias]["text"]:
            raise ValueError("citation原文不属于对应设定")
        seen.add(alias)


def validate_answer(value: dict, evidence: list[dict]) -> None:
    validate_model_answer(value, evidence)
    verdict = value["verdict"]
    assessment = value["assessment"]
    if verdict == "contradiction" and not _contradiction_is_proven(assessment):
        raise ValueError("contradiction必须证明两个命题无法同时为真")
    if verdict == "consistent" and not _support_is_proven(assessment):
        raise ValueError("consistent必须由候选设定直接支持完整事件")


def apply_scope_guard(value: dict) -> dict:
    """Downgrade unsupported explicit verdicts while preserving model output."""
    result = copy.deepcopy(value)
    verdict = result.get("verdict")
    assessment = result.get("assessment")
    if not isinstance(assessment, dict):
        return result
    if verdict == "contradiction" and not _contradiction_is_proven(assessment):
        result["model_verdict"] = verdict
        result["verdict"] = "uncertain"
        result["guard_reason"] = "coexistence_not_ruled_out"
    elif verdict == "consistent" and not _support_is_proven(assessment):
        result["model_verdict"] = verdict
        result["verdict"] = "uncertain"
        result["guard_reason"] = "direct_support_not_proven"
    return result
