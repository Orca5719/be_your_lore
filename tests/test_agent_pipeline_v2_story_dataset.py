import copy
import json
from pathlib import Path
import unittest

from agent_pipeline_v2.story_dataset import validate_story_dataset


ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data" / "index"
PILOT = ROOT / "evaluation" / "story_benchmark_pilot_4_v1.json"
CHUNK_ID = "099d1bbdbc348eb3c332"


def valid_dataset():
    snapshot = json.loads(PILOT.read_text(encoding="utf-8-sig"))["corpus_snapshot"]
    story = (
        "雷独自坐在桌边。他有两颗心脏。窗外阳光落在杯沿上，他端起水杯喝了一口，随后把杯子放回原处。"
        "屋里十分安静，他整理衣袖，又把松开的鞋带重新系好。过了一会儿，他走到窗边看了看街道，再回到椅子旁坐下。"
        "桌角有几粒饼干碎屑，他用纸巾慢慢擦净，把纸巾折好放到碟子旁边，然后靠着椅背休息。"
        "风吹动窗帘，他伸手将窗户掩上一点，确认杯子没有放在桌边，便没有再做别的事。"
        "墙上的钟继续走着，他听了一阵秒针的轻响，又把椅子向桌边挪了少许。"
    )
    fact_text = "他有两颗心脏。"
    ignored_text = "他端起水杯喝了一口，"
    fact_start = story.index(fact_text)
    ignored_start = story.index(ignored_text)
    return {
        "schema_version": "agent-pipeline-v2-story-dataset-v2",
        "name": "test",
        "version": "test-v1",
        "status": "assistant_authored_draft_pending_human_review",
        "notice": "测试设定，不代表正式 canon。",
        "corpus_snapshot": snapshot,
        "cases": [{
            "id": "SL-T01",
            "title": "验证样例",
            "group": "zero_conflict",
            "dimensions": ["physical_rule"],
            "story": story,
            "character_count": len(story),
            "gold_facts": [{
                "id": "G1",
                "subject": "雷",
                "predicate": "拥有",
                "object": "两颗心脏",
                "normalized_fact": "雷拥有两颗心脏",
                "type": "character_attribute",
                "dimension": "physical_rule",
                "source_anchors": [{"text": fact_text, "start": fact_start, "end": fact_start + len(fact_text)}],
                "context_anchors": [],
                "expected_verdict": "一致",
                "relevant_lore_ids": [CHUNK_ID],
                "minimum_evidence_sets": [[CHUNK_ID]],
                "reportable": False,
            }],
            "ignored_events": [{
                "anchor": {"text": ignored_text, "start": ignored_start, "end": ignored_start + len(ignored_text)},
                "normalized_proposition": "雷喝水",
                "reason": "routine",
            }],
            "non_events": [],
            "gold_findings": [],
            "annotation_status": "assistant_authored_draft_pending_human_review",
        }],
    }


class StoryDatasetValidationTests(unittest.TestCase):
    def test_valid_expanded_dataset_reports_counts_and_snapshot(self):
        result = validate_story_dataset(valid_dataset(), ROOT, INDEX, expected_cases=1, expected_facts_per_verdict=None)
        self.assertEqual(result["cases"], 1)
        self.assertEqual(result["gold_facts"], 1)
        self.assertEqual(result["verdicts"], {"一致": 1, "矛盾": 0, "不确定": 0})
        self.assertEqual(result["groups"], {"zero_conflict": 1})

    def test_rejects_inexact_source_anchor(self):
        data = valid_dataset()
        data["cases"][0]["gold_facts"][0]["source_anchors"][0]["start"] += 1
        with self.assertRaisesRegex(ValueError, "金标原文位置"):
            validate_story_dataset(data, ROOT, INDEX, expected_cases=1, expected_facts_per_verdict=None)

    def test_rejects_unknown_lore_and_evidence_outside_relevant_set(self):
        data = valid_dataset()
        data["cases"][0]["gold_facts"][0]["relevant_lore_ids"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "设定片段"):
            validate_story_dataset(data, ROOT, INDEX, expected_cases=1, expected_facts_per_verdict=None)

        data = valid_dataset()
        data["cases"][0]["gold_facts"][0]["minimum_evidence_sets"] = [["47ba19b8b9ece1f35cc9"]]
        with self.assertRaisesRegex(ValueError, "最小证据组合"):
            validate_story_dataset(data, ROOT, INDEX, expected_cases=1, expected_facts_per_verdict=None)

    def test_rejects_duplicate_ids_and_dangling_finding_reference(self):
        data = valid_dataset()
        duplicate = copy.deepcopy(data["cases"][0]["gold_facts"][0])
        data["cases"][0]["gold_facts"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "事实ID重复"):
            validate_story_dataset(data, ROOT, INDEX, expected_cases=1, expected_facts_per_verdict=None)

        data = valid_dataset()
        data["cases"][0]["gold_findings"] = [{
            "id": "F1", "gold_fact_ids": ["missing"], "expected_type": "conflict",
            "dimension": "physical_rule", "minimum_evidence_sets": [[CHUNK_ID]], "merge": False,
        }]
        with self.assertRaisesRegex(ValueError, "不存在的事实"):
            validate_story_dataset(data, ROOT, INDEX, expected_cases=1, expected_facts_per_verdict=None)

    def test_enforces_case_and_verdict_distributions(self):
        with self.assertRaisesRegex(ValueError, "案例数量"):
            validate_story_dataset(valid_dataset(), ROOT, INDEX, expected_cases=24, expected_facts_per_verdict=None)
        with self.assertRaisesRegex(ValueError, "判断分布"):
            validate_story_dataset(valid_dataset(), ROOT, INDEX, expected_cases=1, expected_facts_per_verdict=24)

    def test_group_requires_exact_conflict_count(self):
        data = valid_dataset()
        data["cases"][0]["group"] = "one_conflict"
        with self.assertRaisesRegex(ValueError, "分组与矛盾数量"):
            validate_story_dataset(data, ROOT, INDEX, expected_cases=1, expected_facts_per_verdict=None)


if __name__ == "__main__":
    unittest.main()
