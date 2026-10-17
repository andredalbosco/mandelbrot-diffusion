from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from model import DiT, XPredictionFlow
from mandelbrot import GrayscaleRenderer, ColorRenderer, View
from dataset import load_images


def scalar_coverage(width, height, view, side):
    result = torch.zeros(height, width)
    for y in range(height):
        for x in range(width):
            white = 0
            for sy in range(side):
                for sx in range(side):
                    c = view.center + complex((x+(sx+.5)/side-width/2)*view.span/width,
                                              (height/2-y-(sy+.5)/side)*view.span/width)
                    z = view.z0
                    for _ in range(view.iterations):
                        if abs(z) > 2:
                            break
                        z = z*z + c
                    white += abs(z) > 2
            result[y, x] = white / side**2
    return result


class CoverageTests(unittest.TestCase):
    def test_scalar_subpixel_coverage(self):
        view = View(center=-.7+.1j, span=2.8, iterations=128)
        renderer = GrayscaleRenderer(12, 8, "cuda", side=4)
        torch.testing.assert_close(renderer.render(view).cpu(), scalar_coverage(12, 8, view, 4), rtol=0, atol=0)
        self.assertTrue(((renderer.image > 0) & (renderer.image < 1)).any())
        renderer.render(replace(view, center=0j, span=.01))
        self.assertTrue((renderer.image == 0).all())
        renderer.render(replace(view, center=3+0j, span=.01))
        self.assertTrue((renderer.image == 1).all())

    def test_precision_guard_and_retained_view(self):
        renderer = GrayscaleRenderer(32, 24, "cuda", side=2)
        with self.assertRaises(ValueError):
            renderer.render(View(center=-.75+.1j, span=1e-14))
        view = View(center=-.7+.1j, span=2.8, iterations=128)
        image = renderer.render(view)
        self.assertEqual(image.shape, (24, 32))
        self.assertEqual(renderer.view, view)
        expected = image.clone()
        self.assertEqual(renderer.render().data_ptr(), image.data_ptr())
        torch.testing.assert_close(renderer.image, expected, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            output = renderer.save(Path(directory) / "coverage.png")
            from PIL import Image
            with Image.open(output) as saved:
                self.assertEqual(saved.size, (32, 24))

    def test_resolve_matches_high_resolution_and_budget_is_monotone(self):
        view = View(center=-.75+.1j, span=.2, iterations=1024)
        renderer = GrayscaleRenderer(32, 32, "cuda", side=4)
        low = renderer.render(view).clone()
        reference = ColorRenderer(128, 128, "cuda", view)
        reference.render()
        expected = torch.nn.functional.avg_pool2d((reference.escape != -1).float()[None, None], 4)[0, 0]
        torch.testing.assert_close(low, expected, rtol=0, atol=0)
        high = renderer.render(replace(view, iterations=2048)).clone()
        self.assertTrue((high >= low).all())

    def test_cuda_stream_matches_scalar_reference(self):
        view = View(center=-.7+.1j, span=2.8, iterations=128)
        gpu = GrayscaleRenderer(24, 16, "cuda", side=4)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            result = gpu.render(view).clone()
        stream.synchronize()
        torch.testing.assert_close(result.cpu(), scalar_coverage(24, 16, view, 4), rtol=0, atol=0)


class FlowTests(unittest.TestCase):
    def test_interpolation_and_x_prediction_objective(self):
        flow = XPredictionFlow()
        clean = torch.linspace(-1, 1, 64).reshape(1, 1, 8, 8)
        noise = torch.ones_like(clean)*.25
        t = torch.tensor([.4])
        mixed = flow.interpolate(clean, t, noise)
        torch.testing.assert_close(mixed, .6*clean+.4*noise)
        class Oracle(torch.nn.Module):
            def forward(self, x, t):
                return clean
        self.assertLess(flow.loss(Oracle(), clean, t, noise).item(), 1e-12)
        class Offset(torch.nn.Module):
            def forward(self, x, t):
                return clean+.1
        self.assertAlmostEqual(flow.loss(Offset(), clean, t, noise).item(), (.1/.4)**2, places=6)
        with self.assertRaises(ValueError):
            flow.loss(Oracle(), clean, torch.tensor([0.]), noise)

    def test_sampler_sign_and_endpoints(self):
        flow = XPredictionFlow()
        target = torch.ones(1, 1, 8, 8)*.2
        class ConstantClean(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
            def forward(self, x, t):
                return target.expand_as(x)
        model = ConstantClean()
        for steps in (1, 3, 50):
            torch.testing.assert_close(flow.sample(model, 1, 8, steps=steps), (target+1)/2)

    def test_dataset_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.pt"
            images = torch.tensor([0., .5, 1., 0.]).reshape(1, 1, 2, 2)
            torch.save({"images": images}, path)
            torch.testing.assert_close(load_images(path), images*2-1)
            torch.save({"images": images*float("nan")}, path)
            with self.assertRaises(ValueError):
                load_images(path)


class DiTTests(unittest.TestCase):
    def test_rgb_patches_sampling_and_image_output(self):
        from dataset import save_grid
        from PIL import Image
        model = DiT(size=8, patch_size=2, hidden_size=32, depth=1, num_heads=4, channels=3)
        image = torch.arange(192).reshape(1, 3, 8, 8).float() / 191
        patches = image.reshape(1, 3, 4, 2, 4, 2).permute(0, 2, 4, 3, 5, 1).reshape(1, 16, 12)
        torch.testing.assert_close(model.unpatchify(patches), image, rtol=0, atol=0)
        loss = XPredictionFlow().loss(model, image*2-1)
        loss.backward()
        self.assertTrue(torch.isfinite(model.final_projection.weight.grad).all())
        samples = XPredictionFlow().sample(model, 2, 8, steps=3, channels=3)
        self.assertEqual(samples.shape, (2, 3, 8, 8))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.png"
            save_grid(image, path)
            with Image.open(path) as saved:
                self.assertEqual(saved.mode, "RGB")
                expected = (image[0].permute(1, 2, 0)*255).round().byte().reshape(-1, 3)
                self.assertEqual(saved.tobytes(), expected.numpy().tobytes())
            path = Path(directory) / "rgb.pt"
            torch.save({"images": image}, path)
            torch.testing.assert_close(load_images(path), image*2-1)

    def test_zero_initialization_and_patch_reconstruction(self):
        torch.set_num_threads(2)
        model = DiT(size=16, patch_size=2, hidden_size=32, depth=2, num_heads=4)
        image = torch.arange(256).reshape(1, 1, 16, 16).float()
        patches = image.reshape(1, 1, 8, 2, 8, 2).permute(0, 2, 4, 3, 5, 1).reshape(1, 64, 4)
        torch.testing.assert_close(model.unpatchify(patches), image, rtol=0, atol=0)
        prediction = model(image, torch.tensor([.5]))
        self.assertEqual(prediction.shape, image.shape)
        self.assertTrue((prediction == 0).all())
        tokens = torch.randn(2, 64, 32)
        condition = torch.randn(2, 32)
        torch.testing.assert_close(model.blocks[0](tokens, condition), tokens, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 1, 15, 16), torch.tensor([.5]))
        with self.assertRaises(ValueError):
            DiT(hidden_size=32, num_heads=3)

    def test_dit_learns_and_uses_attention_and_time(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        model = DiT(size=16, patch_size=2, hidden_size=32, depth=2, num_heads=4)
        flow = XPredictionFlow()
        clean = torch.randn(2, 1, 16, 16).clamp(-1, 1)
        t = torch.tensor([.3, .8])
        noise = torch.randn_like(clean)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=0)
        initial = flow.loss(model, clean, t, noise).item()
        for _ in range(25):
            optimizer.zero_grad()
            loss = flow.loss(model, clean, t, noise)
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
        self.assertLess(flow.loss(model, clean, t, noise).item(), initial)
        self.assertGreater(model.blocks[0].attention.qkv.weight.grad.abs().sum().item(), 0)
        a = model(clean, torch.full((2,), .2))
        b = model(clean, torch.full((2,), .8))
        self.assertFalse(torch.equal(a, b))
        sampled = flow.sample(model, 2, 16, steps=10, seed=7)
        torch.testing.assert_close(sampled, flow.sample(model, 2, 16, steps=10, seed=7), rtol=0, atol=0)



if __name__ == "__main__":
    unittest.main()
