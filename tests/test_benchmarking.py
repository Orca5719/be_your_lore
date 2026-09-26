import unittest
from benchmarking import summarize_times

class BenchmarkTests(unittest.TestCase):
    def test_throughput_uses_median_full_corpus_time(self):
        result=summarize_times([4.,1.,2.],48)
        self.assertEqual(result['median_seconds'],2.)
        self.assertEqual(result['texts_per_second'],24.)
        self.assertEqual(result['samples_seconds'],[4.,1.,2.])

    def test_invalid_samples_rejected(self):
        for samples in [[],[0.],[-1.],[float('nan')]]:
            with self.assertRaises(ValueError):
                summarize_times(samples,48)

if __name__=='__main__':
    unittest.main()
