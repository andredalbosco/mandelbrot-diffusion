import tempfile
from pathlib import Path
import unittest

import torch
from torch import nn

from dataset import load_zoom_pairs, zoom_evidence
from model import DiT, JiT, UNet, RectifiedFlow


class ZoomTests(unittest.TestCase):
    def test_zoom_evidence_masks_and_missing_values(self):
        parent = torch.full((3, 3, 16, 16), .25)
        evidence, masks = zoom_evidence(parent, [4, -2, -2], [4, 4, -2])
        torch.testing.assert_close(evidence[0], torch.full_like(evidence[0], .25))
        torch.testing.assert_close(masks[0], torch.ones_like(masks[0]))
        self.assertTrue(.4 < masks[1].mean() < .6)
        self.assertTrue(.2 < masks[2].mean() < .3)
        self.assertTrue((evidence[masks.expand_as(evidence) == 0] == .5).all())

    def test_clean_prediction_flow_and_endpoint(self):
        condition = torch.full((2, 3, 16, 16), -.4)
        target = torch.full_like(condition, .6)

        class ExactClean(torch.nn.Module):
            prediction = "x"

            def forward(self, state, t, y):
                torch.testing.assert_close(y, condition)
                return target

        flow, model = RectifiedFlow(), ExactClean()
        self.assertEqual(flow.loss(model, target, condition, torch.tensor([0., 1.])).item(), 0)
        for steps in (1, 4, 16):
            torch.testing.assert_close(flow.sample(model, condition, steps), (target + 1) / 2)
        self.assertTrue(model.training)

    def test_flow_direction_and_fixed_condition(self):
        condition = torch.full((2, 3, 16, 16), -.4)
        target = torch.full_like(condition, .6)

        class ExactVelocity(nn.Module):
            def forward(self, state, t, y):
                torch.testing.assert_close(y, condition)
                return target - y

        model = ExactVelocity()
        flow = RectifiedFlow()
        self.assertEqual(flow.loss(model, target, condition).item(), 0)
        for steps in (1, 4, 16):
            actual = flow.sample(model, condition, steps)
            torch.testing.assert_close(actual, (target + 1) / 2)
        self.assertTrue(model.training)
        torch.testing.assert_close(condition, torch.full_like(condition, -.4))

    def test_both_architectures_learn_from_condition(self):
        torch.manual_seed(3)
        y = torch.rand(2, 3, 16, 16) * 2 - 1
        target = y.flip(-1)
        t = torch.full((2,), .5)
        for model in (
            DiT(size=16, channels=3, condition_channels=3, hidden_size=32, depth=2, num_heads=4),
            JiT(size=16, hidden_size=32, depth=2, num_heads=4, bottleneck_dim=16),
            UNet(size=16, widths=(8, 16, 32, 32)),
        ):
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            before = RectifiedFlow().loss(model, target, y, t)
            before.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            optimizer.step()
            after = RectifiedFlow().loss(model, target, y, t)
            self.assertLess(after.item(), before.item())
            state = .5 * (y + target)
            self.assertFalse(torch.equal(model(state, t, y), model(state, t, -y)))

    def test_pair_alignment_and_range_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.pt"
            inputs = torch.rand(3, 3, 16, 16)
            targets = inputs.flip(-1)
            torch.save({"inputs": inputs, "targets": targets}, path)
            pairs = load_zoom_pairs(path)
            torch.testing.assert_close(pairs[:, 0], inputs * 2 - 1)
            torch.testing.assert_close(pairs[:, 1], targets * 2 - 1)
            torch.save({"inputs": inputs + 2, "targets": targets}, path)
            with self.assertRaises(ValueError):
                load_zoom_pairs(path)
            torch.save({"inputs": inputs, "targets": targets,
                        "masks": torch.zeros(3, 1, 16, 16)}, path)
            with self.assertRaisesRegex(ValueError, "mask-aware"):
                load_zoom_pairs(path)


if __name__ == "__main__":
    unittest.main()
