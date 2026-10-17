from collections import Counter
import random
import unittest
from unittest.mock import patch

import torch

from dataset import generate_balanced_split, sampling_plan


class SamplingTests(unittest.TestCase):
    def test_marginal_quotas_and_reproducibility(self):
        for count in (100, 900):
            plan = sampling_plan(count, random.Random(3))
            self.assertEqual(len(plan), count)
            self.assertEqual(plan, sampling_plan(count, random.Random(3)))
            coverage = Counter(c for c, s in plan)
            self.assertLessEqual(max(coverage.values())-min(coverage.values()), 1)
            scales = Counter(s for c, s in plan)
            self.assertEqual(scales, {0: round(.1*count), 1: round(.25*count),
                                      2: round(.35*count),
                                      3: count-round(.1*count)-round(.25*count)-round(.35*count)})

    def test_adaptive_budget_retains_difficult_crop(self):
        class Renderer:
            device = torch.device("cpu")
            side = 4
            def render(self, view):
                value = {1024: .75, 2048: .752, 4096: .7521}[view.iterations]
                return torch.full((8, 8), value)
        with patch("mandelbrot.GrayscaleRenderer", return_value=Renderer()):
            images, records, quality = generate_balanced_split(
                1, {"anchor": [(0., 0.)]}, Renderer(), random.Random(4),
                1024, .0005, 4096, set())
        self.assertEqual(images.shape, (1, 1, 8, 8))
        self.assertEqual(records[0]["iterations"], 4096)
        self.assertLess(records[0]["iteration_delta_mean"], .0005)
        self.assertTrue(.2 <= records[0]["black_fraction"] < .3)
        self.assertEqual(quality["attempts"], 1)


if __name__ == "__main__":
    unittest.main()
