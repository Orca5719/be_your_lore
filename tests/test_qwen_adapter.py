import unittest
from qwen_judge import QwenJudge, GenerationError

class QwenAdapterTests(unittest.TestCase):
    def test_truncated_generation_keeps_raw_output_and_error(self):
        judge = QwenJudge.__new__(QwenJudge)
        judge.prompt = 'prompt'
        judge.device = 'cpu'
        def generate(messages, **kwargs):
            raise GenerationError('生成未结束', '{"findings":[')
        judge._generate = generate
        result = judge.check('雷能控制时间。', [])
        self.assertEqual(result['status'], 'error')
        self.assertIsNone(result['overall_verdict'])
        self.assertEqual(result['raw_output'], '{"findings":[')

