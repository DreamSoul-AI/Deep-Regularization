import unittest
import torch
import numpy as np

from experiments.baseline_suite import metrics
from experiments.verify_baseline_evidence import score_metrics


class BaselineMetricTests(unittest.TestCase):
    def test_negative_relief_relevance_is_not_made_important(self):
        values=np.full(128,-10.)
        values[:2]=[1.,2.]
        result=metrics(values,[0,1])
        self.assertEqual(result['top_k'],[1,0])
        self.assertEqual(result['recall_at_k'],1.)
        self.assertEqual(result['average_precision'],1.)
        self.assertEqual(result['auroc'],1.)

    def test_standard_library_metrics_match_sklearn_with_ties(self):
        rng=np.random.default_rng(43)
        for values in [rng.normal(size=128),rng.integers(-2,3,size=128),np.zeros(128)]:
            expected=metrics(values,[1,9,43,67])
            actual=score_metrics(values,[1,9,43,67])
            self.assertEqual(actual['top_k'],expected['top_k'])
            for name in ['recall_at_k','average_precision','auroc']:
                self.assertAlmostEqual(actual[name],expected[name],places=12)


if __name__=='__main__':
    unittest.main()
