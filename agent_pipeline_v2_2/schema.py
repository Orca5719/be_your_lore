from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any


STATUSES = frozenset({"ok", "partial", "error", "pending_review"})
STAGES = ("extraction", "retrieval", "judge", "report", "total")


def stable_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SystemConfig:
    name: str
    retrieval: str
    judge: str
    metadata_filter: bool
    top_k: int
    judge_batch_size: int

    @classmethod
    def baseline(cls) -> "SystemConfig":
        return cls("baseline", "dense", "v1", False, 5, 8)

    @classmethod
    def candidate(cls) -> "SystemConfig":
        return cls("candidate", "hybrid", "v2.1", False, 5, 8)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_system_config(value: SystemConfig) -> SystemConfig:
    if not isinstance(value, SystemConfig):
        raise ValueError("system config must be SystemConfig")
    expected = SystemConfig.baseline() if value.name == "baseline" else SystemConfig.candidate() if value.name == "candidate" else None
    if expected is None:
        raise ValueError("unknown system name")
    if value.metadata_filter:
        raise ValueError("metadata filter must be disabled")
    if value.top_k != 5:
        raise ValueError("Top-K must be 5")
    if value.judge_batch_size != 8:
        raise ValueError("Judge batch must be 8")
    if (value.retrieval, value.judge) != (expected.retrieval, expected.judge):
        raise ValueError("retrieval/Judge configuration does not match the named system")
    return value


@dataclass(frozen=True)
class RunIdentity:
    dataset_sha256: str
    config_sha256: str
    case_ids: tuple[str, ...]

    @classmethod
    def create(cls, dataset_sha256: str, config: SystemConfig, case_ids: list[str]) -> "RunIdentity":
        validate_system_config(config)
        if not isinstance(dataset_sha256, str) or not dataset_sha256:
            raise ValueError("dataset digest is required")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("duplicate case IDs")
        if not case_ids or any(not isinstance(x, str) or not x for x in case_ids):
            raise ValueError("case IDs must be non-empty strings")
        return cls(dataset_sha256, stable_digest(config.to_dict()), tuple(case_ids))


def validate_run_row(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("run row must be an object")
    if not isinstance(value.get("case_id"), str) or not value["case_id"]:
        raise ValueError("case_id is required")
    if value.get("system") not in {"baseline", "candidate"}:
        raise ValueError("system is invalid")
    if value.get("status") not in STATUSES:
        raise ValueError("status is invalid")
    if not isinstance(value.get("result"), dict):
        raise ValueError("result must be an object")
    metrics = value.get("stage_metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(STAGES):
        raise ValueError("stage metrics must contain extraction, retrieval, judge, report, and total")
    for name in STAGES:
        seconds = metrics[name].get("seconds") if isinstance(metrics[name], dict) else None
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"stage metrics {name} seconds must be finite and non-negative")
    return value
