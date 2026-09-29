import unittest
from evaluate import score_case

class EvaluationTests(unittest.TestCase):
    def test_core_hit_does_not_hide_missing_additional(self):
        row=score_case('core',['extra'],[{'id':'core'},{'id':'other'}])
        self.assertTrue(row['core_hit'])
        self.assertEqual(row['core_rank'],1)
        self.assertEqual(row['evidence_recall'],0.5)
        self.assertEqual(row['missing_ids'],['extra'])

    def test_additional_hit_is_not_core_hit(self):
        row=score_case('core',['extra'],[{'id':'extra'}])
        self.assertFalse(row['core_hit'])
        self.assertIsNone(row['core_rank'])
        self.assertEqual(row['evidence_recall'],0.5)

if __name__=='__main__':
    unittest.main()
