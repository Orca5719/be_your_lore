import unittest
from pathlib import Path

from agent_pipeline_v2.story_dataset import load_story_dataset


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evaluation" / "story_benchmark_24_v2.json"
INDEX = ROOT / "data" / "index"


class RealStoryDatasetTests(unittest.TestCase):
    def test_real_dataset_has_confirmed_shape(self):
        data, summary = load_story_dataset(DATASET, ROOT, INDEX)
        self.assertEqual(summary["cases"], 24)
        self.assertEqual(summary["gold_facts"], 72)
        self.assertEqual(summary["verdicts"], {"一致": 24, "矛盾": 24, "不确定": 24})
        self.assertEqual(summary["groups"], {"zero_conflict": 8, "one_conflict": 8, "multi_conflict": 8})
        self.assertEqual(len(data["cases"]), 24)

    def test_every_one_conflict_dimension_is_represented(self):
        _, summary = load_story_dataset(DATASET, ROOT, INDEX)
        self.assertGreaterEqual(summary["dimensions"].get("time", 0), 1)
        self.assertGreaterEqual(summary["dimensions"].get("character_knowledge", 0), 1)
        self.assertGreaterEqual(summary["dimensions"].get("space", 0), 1)
        self.assertGreaterEqual(summary["dimensions"].get("character_relation", 0), 1)
        self.assertGreaterEqual(summary["dimensions"].get("physical_rule", 0), 1)
        self.assertGreaterEqual(summary["dimensions"].get("world_rule", 0), 1)
        self.assertGreaterEqual(summary["dimensions"].get("causality", 0), 1)
        self.assertGreaterEqual(summary["dimensions"].get("identity", 0), 1)


if __name__ == "__main__":
    unittest.main()
