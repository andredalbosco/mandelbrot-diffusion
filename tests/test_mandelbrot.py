import math
from decimal import Decimal, localcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from mandelbrot import ColorRenderer, View
from viewer import Viewer, zoom_view, pan_view


def reference(width, height, view):
    counts = torch.empty((height, width), dtype=torch.int32)
    smooth = torch.full((height, width), -1.0)
    step = view.span / width
    for y in range(height):
        for x in range(width):
            c = view.center + complex((x + 0.5 - width / 2) * step,
                                      (height / 2 - y - 0.5) * step)
            z = view.z0
            n = 0
            while n < view.iterations and z.real * z.real + z.imag * z.imag <= 4:
                z = z * z + c
                n += 1
            counts[y, x] = n
            radius_squared = z.real * z.real + z.imag * z.imag
            if radius_squared > 4:
                refined_n = n
                for _ in range(3):
                    if radius_squared < 1e100:
                        z = z * z + c
                        radius_squared = z.real * z.real + z.imag * z.imag
                        refined_n += 1
                smooth[y, x] = refined_n + 1 - math.log(0.5 * math.log(radius_squared)) / math.log(2)
    return counts, smooth


class RendererTests(unittest.TestCase):
    def test_color_supersampling_averages_colored_subpixels(self):
        view = View(center=-.75+.1j, span=.2, iterations=512)
        low = ColorRenderer(16, 12, side=2)
        high = ColorRenderer(32, 24)
        low.render(view)
        high.render(view)
        self.assertEqual(low.image.shape, (12, 16, 3))
        self.assertEqual(low.counts.shape, (24, 32))
        for style in ("cosine", "3b1b", "fire", "grayscale"):
            actual = low.recolor(style=style)
            samples = high.recolor(style=style)
            expected = torch.nn.functional.avg_pool2d(samples.permute(2, 0, 1)[None], 2)[0].permute(1, 2, 0)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        with self.assertRaises(ValueError):
            ColorRenderer(16, 12, side=0)

    def test_scalar_reference(self):
        renderer = ColorRenderer(32, 24, "cuda")
        for view in (View(center=-0.55+0.08j, span=2.6, iterations=80),
                     View(center=-0.3+0.2j, span=3.2, iterations=40, z0=0.2+0.1j)):
            with self.subTest(view=view):
                rgb = renderer.render(view)
                counts, smooth = reference(32, 24, view)
                self.assertTrue(torch.equal(renderer.counts.cpu(), counts))
                torch.testing.assert_close(renderer.escape.cpu(), smooth, rtol=1e-5, atol=1e-5)
                self.assertEqual(rgb.shape, (24, 32, 3))
                self.assertTrue(torch.isfinite(rgb).all())
                self.assertTrue(((rgb >= 0) & (rgb <= 1)).all())
                self.assertTrue((rgb.cpu()[smooth == -1] == 0).all())

    def test_known_orbits_and_last_iteration_escape(self):
        renderer = ColorRenderer(1, 1, "cuda")
        for center, iterations, count, escaped in ((0j, 16, 16, False), (-1+0j, 16, 16, False),
                                                   (-2+0j, 16, 16, False), (2+0j, 1, 1, False),
                                                   (2+0j, 2, 2, True), (3+0j, 16, 1, True)):
            renderer.render(View(center=center, iterations=iterations))
            self.assertEqual(renderer.counts.item(), count)
            self.assertEqual(renderer.escape.item() != -1, escaped)
        renderer.render(View(center=0j, iterations=16, z0=3+0j))
        self.assertEqual(renderer.counts.item(), 0)
        self.assertNotEqual(renderer.escape.item(), -1)

    def test_reused_shared_buffers(self):
        renderer = ColorRenderer(8, 6, "cuda")
        pointers = [tensor.data_ptr() for tensor in (renderer.counts, renderer.escape, renderer.image)]
        self.assertEqual(pointers, [array.ptr for array in renderer._outputs])
        first = renderer.render().clone()
        renderer.render(View(center=2+0j))
        self.assertEqual(pointers[-1], renderer.image.data_ptr())
        self.assertFalse(torch.equal(first, renderer.image))

    def test_palette_reuses_orbits(self):
        renderer = ColorRenderer(32, 24, "cuda")
        first = renderer.render().clone()
        counts, smooth = renderer.counts.clone(), renderer.escape.clone()
        changed = renderer.recolor(period=20, phase=0.25).clone()
        self.assertFalse(torch.equal(first, changed))
        self.assertTrue(torch.equal(counts, renderer.counts))
        self.assertTrue(torch.equal(smooth, renderer.escape))
        self.assertTrue((changed[smooth == -1] == 0).all())
        torch.testing.assert_close(renderer.recolor(phase=1.25), changed, rtol=1e-4, atol=1e-5)
        with self.assertRaises(ValueError):
            renderer.recolor(period=0)

    def test_color_options_and_3b1b_mapping(self):
        renderer = ColorRenderer(32, 24)
        renderer.render(View(iterations=80))
        counts, escape = renderer.counts.clone(), renderer.escape.clone()
        for style in ("3b1b", "fire", "grayscale"):
            for mapping in ("linear", "sqrt", "log"):
                for repeat in (False, True):
                    image = renderer.recolor(style=style, mapping=mapping, repeat=repeat)
                    self.assertTrue(torch.isfinite(image).all())
                    self.assertTrue(((image >= 0) & (image <= 1)).all())
                    self.assertTrue((image[escape == -1] == 0).all())
                    self.assertTrue(torch.equal(counts, renderer.counts))
                    self.assertTrue(torch.equal(escape, renderer.escape))
                    if style == "grayscale":
                        torch.testing.assert_close(image[:, :, 0], image[:, :, 1])
                        torch.testing.assert_close(image[:, :, 1], image[:, :, 2])
        # A one-pixel orbit c=3 escapes on the first update (zero-based n=0).
        single = ColorRenderer(1, 1)
        single.render(View(center=3+0j))
        expected = math.log(2) / math.log(3) + 1.5
        self.assertAlmostEqual(single.styled_escape.item(), expected, places=6)
        image = single.recolor(style="3b1b", mapping="sqrt", repeat=False)
        alpha = max(0, min(1, (math.sqrt(expected)-1.5)/6.5))
        position = alpha * 8
        index = min(int(position), 7)
        palette = single.palettes["3b1b"]
        color = torch.lerp(palette[index], palette[index+1], position-index)
        torch.testing.assert_close(image[0, 0], color)
        for kwargs in ({"lower": 8, "upper": 1}, {"mapping": "bad"}, {"style": "bad"}):
            with self.assertRaises(ValueError):
                single.recolor(**kwargs)

    def test_input_validation(self):
        for kwargs in ({"span": 0}, {"center": complex(float("nan"), 0)},
                       {"iterations": 0}, {"iterations": 1.5}, {"z0": complex(float("inf"), 0)}):
            with self.assertRaises(ValueError):
                View(**kwargs)
        with self.assertRaises(ValueError):
            ColorRenderer(0, 8)
        with self.assertRaises(ValueError):
            ColorRenderer(device="mps")

    def test_cuda_and_current_stream(self):
        view = View(center=-0.55+0.08j, span=2.6, iterations=80)
        expected_counts, expected_smooth = reference(32, 24, view)
        gpu = ColorRenderer(32, 24, "cuda:0")
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            gpu.image.zero_()
            rgb = gpu.render(view)
            consumed = rgb * 2
            counts = gpu.counts.clone()
            smooth = gpu.escape.clone()
        stream.synchronize()
        self.assertTrue(torch.equal(counts.cpu(), expected_counts))
        torch.testing.assert_close(smooth.cpu(), expected_smooth, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(consumed, gpu.image * 2, rtol=1e-5, atol=1e-5)


class NavigationTests(unittest.TestCase):
    def test_zoom_preserves_anchor_and_respects_precision(self):
        view = View(center=-.5+0j)
        anchor = -.75+.1j
        zoomed = zoom_view(view, anchor, .8, 1e-12)
        self.assertAlmostEqual(zoomed.span, view.span*.8)
        self.assertAlmostEqual((anchor-zoomed.center)/zoomed.span,
                               (anchor-view.center)/view.span)
        self.assertEqual(zoom_view(view, anchor, 1e-20, 1e-12).span, 1e-12)

    def test_pan_uses_image_coordinates(self):
        view = View()
        moved = pan_view(view, 20, 10, 1000)
        self.assertAlmostEqual(moved.center.real, view.center.real - 20*view.span/1000)
        self.assertAlmostEqual(moved.center.imag, view.center.imag + 10*view.span/1000)


class ViewerStateTests(unittest.TestCase):
    def test_restore_view_palette_and_window_geometry(self):
        window = SimpleNamespace(screen=SimpleNamespace(width=3840, height=2160),
                                 get_location=lambda: (20, 30), get_size=lambda: (1200, 800),
                                 set_size=Mock(), set_location=Mock())
        viewer = Viewer.__new__(Viewer)
        viewer.gradient_mode = True
        viewer.gradient_auto_contrast = True
        viewer.gradient_scale = 7.
        viewer.window_renderer = SimpleNamespace(window=window)
        viewer.renderer = ColorRenderer(16, 12)
        view = View(center="(-2.0000000000000000000000001+0j)", span=1e-24, iterations=128)
        viewer.renderer.render(view)
        viewer.renderer.recolor(style="fire", mapping="log", phase=.2, lower=.5, upper=6, repeat=True)
        expected = viewer.renderer.image.clone()
        with tempfile.TemporaryDirectory() as directory:
            viewer.state_path = Path(directory) / "state.json"
            viewer.save_state()
            viewer.renderer = ColorRenderer(16, 12)
            viewer.gradient_mode = False
            viewer.gradient_scale = 10.
            viewer.load_state()
            self.assertTrue(viewer.gradient_mode)
            self.assertEqual(viewer.gradient_scale,7.)
        self.assertEqual(viewer.renderer.view, view)
        self.assertEqual(viewer.renderer.view.center_exact, view.center_exact)
        self.assertEqual(viewer.renderer.style, "fire")
        self.assertEqual(viewer.renderer.mapping, "log")
        self.assertEqual(viewer.renderer.phase, .2)
        self.assertTrue(viewer.renderer.repeat)
        torch.testing.assert_close(viewer.renderer.image, expected)
        window.set_size.assert_called_once_with(1200, 800)
        window.set_location.assert_called_once_with(20, 30)


class DeepZoomTests(unittest.TestCase):
    def test_deep_render_matches_60_digit_reference(self):
        view = View(center="(-2.0000000000000000000000001+0j)", span=1e-24, iterations=128)
        renderer = ColorRenderer(16, 12)
        renderer.render(view)
        expected = torch.empty((12, 16), dtype=torch.int32)
        with localcontext() as context:
            context.prec = 60
            step = Decimal(str(view.span))/16
            for y in range(12):
                for x in range(16):
                    cx = Decimal(view.center_exact[0]) + (Decimal(x)+Decimal('.5')-8)*step
                    cy = Decimal(view.center_exact[1]) + (6-Decimal(y)-Decimal('.5'))*step
                    zx = zy = Decimal(0)
                    n = 0
                    while n < view.iterations and zx*zx+zy*zy <= 4:
                        zx, zy = zx*zx-zy*zy+cx, 2*zx*zy+cy
                        n += 1
                    expected[y, x] = n
        self.assertTrue(torch.equal(renderer.counts.cpu(), expected))
        self.assertGreater(renderer.counts.unique().numel(), 3)
        self.assertTrue(renderer.image.isfinite().all())
        with self.assertRaises(ValueError):
            renderer.render(View(span=1e-40))

    def test_navigation_and_text_keep_sub_float64_offsets(self):
        view = View(center="(-0.7436438870371510000000001+0.1318259042053300000000002j)", span=1e-24)
        moved = pan_view(view, 10, 5, 1000)
        self.assertEqual(view.center, moved.center)
        self.assertNotEqual(view.center_exact, moved.center_exact)
        restored = View(center=moved.center_text, span=moved.span)
        self.assertEqual(restored.center_exact, moved.center_exact)
        zoomed = zoom_view(view, view.center, .5, 1e-28, offset=complex(1e-25, 0))
        self.assertNotEqual(view.center_exact, zoomed.center_exact)
        self.assertEqual(zoomed.span, 5e-25)


if __name__ == "__main__":
    unittest.main()
