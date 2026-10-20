"""One-time provenance script for the assistant's 4.1 source-span audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .audit import event_signature


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation/story_benchmark_24_v2.json"
SOURCE = ROOT / "agent_pipeline_v4/reports/benchmark_4B_20261004T132545Z_0fdce1"
B3 = ROOT / "agent_pipeline_v3/reports/benchmark_3_20260927T142334Z_3ceb30/story_runs.jsonl"
OLD = ROOT / "agent_pipeline_v2_2/reports/benchmark_2_2_20260926T122502Z_ecbff6/review.json"


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# Changes found by reading the cited source span and the gold proposition for all
# 4B events. A key uses story and semantic event content, never E1/E2 alone.
CORRECTIONS = {
    ("SL-P02", "雷的生理结构包含两颗心脏，且分别与亚巴顿和拉古艾尔存在宿主关系"):
        ("duplicate", [], "duplicates G1–G3 as a composite"),
    ("SL-P03", "雷拥有两颗心脏"):
        ("hallucinated", [], "right-side angel occupancy does not entail two hearts"),
    ("SL-005", "雷拥有两颗心脏"):
        None,  # E1 valid; E4 source-specific correction below.
    ("SL-006", "雷拥有两颗心脏"):
        ("hallucinated", [], "right-side angel occupancy does not entail two hearts"),
    ("SL-008", "雷拥有两颗心脏"):
        ("valid_checkable", [], "double-heart secret supports anatomy, not disclosure to Delta"),
    ("SL-012", "雷拥有两颗心脏"):
        ("valid_checkable", [], "double-heart secret supports anatomy, not disclosure to Delta"),
    ("SL-020", "雷和德尔塔已经当面交谈过"):
        ("valid_checkable", [], "event omits first-episode timing required by G1"),
    ("SL-022", "雷拥有两颗心脏"):
        ("hallucinated", [], "left-heart angel occupancy does not entail two hearts"),
    ("SL-023", "雷拥有两颗心脏"):
        ("hallucinated", [], "right-heart angel occupancy does not entail two hearts"),
    ("SL-013", "雷拥有永久不死之身"):
        None,  # E1 valid; E4 source-specific correction below.
    ("SL-014", "雷拥有让已死亡的人复活的能力"):
        ("valid_checkable", ["G2"], "directly stated by S5"),
    ("SL-014", "雷与德尔塔在第一话正式见面"):
        ("valid_checkable", ["G3"], "directly stated by S6"),
    ("SL-022", "亚巴顿寄宿在雷的左侧心脏里"):
        ("valid_checkable", ["G3"], "directly stated by S6"),
    ("SL-020", "月城在地震前已经使用机械左臂"):
        ("valid_checkable", ["G4"], "directly stated by S7"),
    ("SL-017", "公众知道德尔塔的真实姓名"):
        ("valid_checkable", ["G1"], "directly stated by S4"),
    ("SL-024", "月城在2064年完成机械左臂安装"):
        ("valid_checkable", ["G3"], "directly stated by S6"),
}


def main():
    old = json.loads(OLD.read_text(encoding="utf-8"))
    baseline = {row["case_id"]: row for row in read_rows(B3)}
    result = {"schema_version": "benchmark-4.1-review-v1", "provenance": "assistant-reviewed",
              "basis": "story source spans and gold propositions; prior 2.2 labels used only for exact semantic matches and rechecked",
              "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
              "source_sha256": {}, "capacities": {}}
    for capacity in (2, 4, 8):
        source_path = SOURCE / f"capacity_{capacity}" / "extractor_runs.jsonl"
        result["source_sha256"][str(capacity)] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        ledger = {}
        for row in read_rows(source_path):
            cid = row["case_id"]
            old_events = baseline[cid]["result"]["benchmark_stages"]["extraction"]["events"]
            old_by_sig = {event_signature(event): event for event in old_events}
            old_labels = old["system_event_labels"].get("candidate:" + cid, {})
            old_mapping = old["event_mapping"].get("candidate:" + cid, {})
            events = row["result"]["benchmark_stages"]["extraction"]["events"]
            labels = {}
            for event in events:
                signature = event_signature(event)
                original = old_by_sig.get(signature)
                old_id = original["id"] if original else None
                previous = old_labels.get(old_id, {}) if old_id else {}
                label = previous.get("label", "pending")
                gold_ids = [gold for gold, ids in old_mapping.items() if old_id in ids] if old_id else []
                note = "reviewed against original source span and gold"
                special = CORRECTIONS.get((cid, event["event"]))
                if special:
                    label, gold_ids, note = special
                if (cid, event["event"]) == ("SL-005", "雷拥有两颗心脏") and event["id"] == "E4":
                    label, gold_ids, note = "hallucinated", [], "E4 cites an unrelated room-description span"
                if (cid, event["event"]) == ("SL-013", "雷拥有永久不死之身") and event["id"] == "E4":
                    label, gold_ids, note = "hallucinated", [], "E4 cites sleeve-adjusting, while E1 already extracts the fact"
                if label == "pending":
                    raise ValueError(f"unreviewed changed event: CB{capacity}/{cid}/{event['id']}: {event['event']}")
                if label != "valid_checkable":
                    gold_ids = []
                labels[event["id"]] = {"signature": signature, "label": label,
                                       "gold_ids": gold_ids, "note": note}
            ledger[cid] = labels
        result["capacities"][str(capacity)] = ledger
    output = ROOT / "agent_pipeline_v4_1/review.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
