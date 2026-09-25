import unittest

from agent_pipeline_v2.fixtures import build_fixture, select_claims, validate_fixture


def dataset(per_label=3):
    cases = []
    labels = [("consistent", "一致"), ("contradiction", "矛盾"), ("insufficient", "不确定")]
    for group, verdict in labels:
        for number in range(per_label):
            quote = f"{verdict}事实{number}"
            evidence = [] if verdict == "不确定" else [{"file": "lore.md", "start_line": number + 1, "end_line": number + 1, "quote": f"设定{number}"}]
            cases.append({
                "id": f"{group}-{number}",
                "group": group,
                "primary_dimension": f"维度{number % 2}",
                "family_id": f"family-{number}",
                "text": quote + "。",
                "prior_development_semantic_overlap": False,
                "claims": [{"input_quote": quote, "expected_verdict": verdict, "evidence": evidence, "subject_any": ["雷"]}],
            })
    return {"name": "source", "cases": cases}


def chunks(count=5):
    return [{
        "id": f"C{number}",
        "text": f"设定{number}",
        "file": "lore.md",
        "start_line": number + 1,
        "end_line": number + 1,
        "heading_path": ["测试"],
    } for number in range(count)]


class Retriever:
    def __init__(self, rows):
        self.rows = rows

    def search(self, query, k=5):
        return [dict(row, score=1 - index * 0.1) for index, row in enumerate(self.rows[:k])]


class V2FixtureTests(unittest.TestCase):
    def test_selection_is_balanced_unique_and_deterministic(self):
        source = dataset(3)
        first = select_claims(source, per_label=2)
        second = select_claims(source, per_label=2)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual({label: sum(row["expected_verdict"] == label for row in first) for label in ("一致", "矛盾", "不确定")}, {"一致": 2, "矛盾": 2, "不确定": 2})
        self.assertEqual(len({row["fixture_id"] for row in first}), 6)

    def test_fixture_includes_canonical_evidence_for_explicit_labels(self):
        fixture = build_fixture(dataset(2), {"chunks": chunks(), "version": "idx"}, Retriever(chunks()), per_label=1, k=3)
        validate_fixture(fixture, expected_per_label=1, expected_k=3)
        for item in fixture["items"]:
            if item["expected_verdict"] != "不确定":
                self.assertTrue(item["canonical_evidence_ids"])
                self.assertTrue(set(item["canonical_evidence_ids"]) <= {row["id"] for row in item["lore"]})

    def test_development_overlap_cases_are_excluded(self):
        source = dataset(2)
        source["cases"][0]["prior_development_semantic_overlap"] = True
        selected = select_claims(source, per_label=1)
        self.assertNotEqual(selected[0]["source_case_id"], source["cases"][0]["id"])


if __name__ == "__main__":
    unittest.main()
