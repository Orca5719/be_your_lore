import unittest
from model_metrics import TokenTiming, metric_baseline

class MetricTests(unittest.TestCase):
    def test_prompt_is_not_first_generated_token(self):
        values=iter([10.2,10.4,10.6])
        timing=TokenTiming(clock=lambda:next(values))
        timing.put(None)
        timing.put(None)
        timing.put(None)
        timing.put(None)
        result=timing.metrics(10.0)
        self.assertAlmostEqual(result['ttft_ms'],200)
        self.assertEqual(result['decode_tokens'],2)
        self.assertAlmostEqual(result['decode_seconds'],0.4)
        timing=TokenTiming(clock=lambda:10.2)
        timing.put(None);timing.put(None)
        self.assertIsNone(timing.metrics(10.0)['decode_tokens_per_second'])

    def test_aggregate_decode_excludes_prefill_and_cold_request(self):
        rows=[{'seconds':20,'generations':[{'input_tokens':100,'generated_tokens':3,'seconds':2,'ttft_ms':500,'decode_tokens':2,'decode_seconds':1}]},
              {'seconds':4,'generations':[{'input_tokens':200,'generated_tokens':5,'seconds':2,'ttft_ms':300,'decode_tokens':4,'decode_seconds':0.5}]}]
        report={'model':'Qwen','totals':{'verdict_correct':1,'claims_total':2},'device':'cuda','peak_allocated_mib':1024}
        result=metric_baseline(report,rows)
        self.assertEqual(result['accuracy'],0.5)
        self.assertEqual(result['avg_input_tokens'],150)
        self.assertEqual(result['warm_total_latency_median_seconds'],4)
        self.assertEqual(result['decode_tokens_per_second'],4)
        self.assertIsNone(result['model_footprint_gib'])
