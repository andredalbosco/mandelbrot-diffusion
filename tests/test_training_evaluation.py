import unittest
from collections import Counter
from scripts.evaluate_training import select_views, MODES


class EvaluationTests(unittest.TestCase):
    def test_depth_balance_unique_and_reproducible(self):
        records = [dict(mode=mode, zoom_depth=depth) for mode in MODES
                   for depth in range(13) for _ in range(7)]
        ids = select_views(records, per_mode=64, seed=17)
        self.assertEqual(len(ids), 256)
        self.assertEqual(len(set(ids)), 256)
        self.assertEqual(ids, select_views(records, per_mode=64, seed=17))
        for mode in MODES:
            counts = Counter(records[i]['zoom_depth'] for i in ids if records[i]['mode'] == mode)
            self.assertEqual(sum(counts.values()), 64)
            self.assertLessEqual(max(counts.values())-min(counts.values()), 1)

    def test_sparse_depths_terminate_without_duplicate_ids(self):
        records = [dict(mode='interior', zoom_depth=12), dict(mode='edge', zoom_depth=1)]
        self.assertEqual(select_views(records), [0, 1])
