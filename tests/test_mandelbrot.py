import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import warp as wp

from mandelbrot import Mandelbrot, View, Viewer


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
                smooth[y, x] = n + 1 - math.log(0.5 * math.log(radius_squared)) / math.log(2)
    return counts, smooth


class RendererTests(unittest.TestCase):
    def test_scalar_reference(self):
        renderer = Mandelbrot(32, 24, "cpu")
        for view in (View(center=-0.55+0.08j, span=2.6, iterations=80),
                     View(center=-0.3+0.2j, span=3.2, iterations=40, z0=0.2+0.1j)):
            with self.subTest(view=view):
                rgb = renderer.render(view)
                counts, smooth = reference(32, 24, view)
                self.assertTrue(torch.equal(renderer.counts, counts))
                torch.testing.assert_close(renderer.escape, smooth, rtol=1e-5, atol=1e-5)
                self.assertEqual(rgb.shape, (24, 32, 3))
                self.assertTrue(torch.isfinite(rgb).all())
                self.assertTrue(((rgb >= 0) & (rgb <= 1)).all())
                self.assertTrue((rgb[smooth == -1] == 0).all())

    def test_known_orbits_and_last_iteration_escape(self):
        renderer = Mandelbrot(1, 1, "cpu")
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
        renderer = Mandelbrot(8, 6, "cpu")
        pointers = [tensor.data_ptr() for tensor in (renderer.counts, renderer.escape, renderer.image)]
        self.assertEqual(pointers, [array.ptr for array in renderer._outputs])
        first = renderer.render().clone()
        renderer.render(View(center=2+0j))
        self.assertEqual(pointers[-1], renderer.image.data_ptr())
        self.assertFalse(torch.equal(first, renderer.image))

    def test_input_validation(self):
        for kwargs in ({"span": 0}, {"center": complex(float("nan"), 0)},
                       {"iterations": 0}, {"iterations": 1.5}, {"z0": complex(float("inf"), 0)}):
            with self.assertRaises(ValueError):
                View(**kwargs)
        with self.assertRaises(ValueError):
            Mandelbrot(0, 8)
        with self.assertRaises(ValueError):
            Mandelbrot(device="mps")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_and_current_stream(self):
        view = View(center=-0.55+0.08j, span=2.6, iterations=80)
        cpu = Mandelbrot(32, 24, "cpu")
        cpu.render(view)
        gpu = Mandelbrot(32, 24, "cuda:0")
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            gpu.image.zero_()
            rgb = gpu.render(view)
            consumed = rgb * 2
            counts = gpu.counts.clone()
            smooth = gpu.escape.clone()
        stream.synchronize()
        self.assertTrue(torch.equal(counts.cpu(), cpu.counts))
        torch.testing.assert_close(smooth.cpu(), cpu.escape, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(consumed.cpu(), cpu.image * 2, rtol=1e-5, atol=1e-5)


class ViewerTests(unittest.TestCase):
    def test_controls(self):
        renderer = Mandelbrot(32, 24, "cpu", View(iterations=32))
        viewer = Viewer(renderer)
        try:
            viewer.figure.canvas.draw()
            initial = renderer.view
            anchor = -0.75+0.1j
            viewer.on_scroll(SimpleNamespace(inaxes=viewer.axes, xdata=anchor.real, ydata=anchor.imag, step=1))
            self.assertAlmostEqual(renderer.view.span, initial.span / 1.25)
            self.assertAlmostEqual((anchor - renderer.view.center) / renderer.view.span,
                                   (anchor - initial.center) / initial.span)
            x, y = viewer.axes.bbox.x0 + 50, viewer.axes.bbox.y0 + 50
            before_pan = renderer.view
            viewer.on_press(SimpleNamespace(inaxes=viewer.axes, button=1, x=x, y=y))
            viewer.on_motion(SimpleNamespace(inaxes=viewer.axes, x=x+20, y=y+10))
            self.assertLess(renderer.view.center.real, before_pan.center.real)
            self.assertLess(renderer.view.center.imag, before_pan.center.imag)
            viewer.on_release(None)
            self.assertIsNone(viewer.drag)
            viewer.fields["z0"].set_val("0.2+0.1j")
            self.assertEqual(renderer.view.z0, 0.2+0.1j)
            valid = renderer.view
            viewer.fields["span"].set_val("bad")
            self.assertEqual(renderer.view, valid)
            viewer.on_reset()
            self.assertEqual(renderer.view, initial)
            with tempfile.TemporaryDirectory() as directory:
                viewer.output = str(Path(directory) / "saved.png")
                viewer.on_key(SimpleNamespace(key="s"))
                saved = plt.imread(viewer.output)
                torch.testing.assert_close(torch.from_numpy(saved[:, :, :3]), renderer.image, rtol=0, atol=1/255)
        finally:
            plt.close(viewer.figure)


if __name__ == "__main__":
    unittest.main()
