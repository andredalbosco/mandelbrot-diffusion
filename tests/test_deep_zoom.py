"""Independent high-precision checks for perturbation and its direct fallback."""
from decimal import Decimal, localcontext
import unittest
from unittest.mock import patch

import torch
import warp as wp

from mandelbrot import ColorRenderer, View


def decimal_counts(width, height, view):
    result = torch.empty((height, width), dtype=torch.int32)
    with localcontext() as context:
        context.prec = 70
        step = Decimal(str(view.span))/width
        for y in range(height):
            for x in range(width):
                cx = Decimal(view.center_exact[0])+(Decimal(x)+Decimal('.5')-Decimal(width)/2)*step
                cy = Decimal(view.center_exact[1])+(Decimal(height)/2-Decimal(y)-Decimal('.5'))*step
                zx, zy = Decimal.from_float(view.z0.real), Decimal.from_float(view.z0.imag)
                n = 0
                while n < view.iterations and zx*zx+zy*zy <= 4:
                    zx, zy = zx*zx-zy*zy+cx, 2*zx*zy+cy
                    n += 1
                result[y, x] = n
    return result


class PerturbationTests(unittest.TestCase):
    def test_70_digit_escape_counts(self):
        renderer = ColorRenderer(12, 8)
        views = [
            View(center='(-2.0000000000000000000000001+0j)', span=1e-24, iterations=128),
            View(center='(-0.92844118559550850366513868844580424143709174+0.262617371450439604741270010095574502976117479j)', span=4.118046071574424e-20),
            View(center='(-0.7436438870371510000000001+0.1318259042053300000000002j)', span=1e-16, iterations=2048),
            View(center=0j, span=1e-18, iterations=40),
            View(center=3+0j, span=1e-18, iterations=40),
            View(center=0j, span=1e-18, iterations=40, z0=3+0j),
            View(center=0j, span=1e-18, iterations=40, z0=2+0j),
            View(center='(-0.3+0.2j)', span=1e-18, iterations=80, z0=.2+.1j),
        ]
        for view in views:
            with self.subTest(view=view):
                renderer.render(view)
                self.assertTrue(torch.equal(renderer.counts.cpu(), decimal_counts(12, 8, view)))
                self.assertTrue(torch.isfinite(renderer.image).all())

    def test_fast_path_and_uncertain_bailout_fallback_are_both_exercised(self):
        renderer = ColorRenderer(12, 8)
        launch = wp.launch
        fallback_counts = []
        def track_launch(kernel, *args, **kwargs):
            if kernel.key == 'batched_dd_fallback':
                fallback_counts.append(int(renderer._tiled_backend.queue_length.numpy()[0]))
            return launch(kernel, *args, **kwargs)
        with patch.object(wp, 'launch', track_launch):
            renderer.render(View(center='(-0.7436438870371510000000001+0.1318259042053300000000002j)', span=1e-16))
            renderer.render(View(center='(-2.0000000000000000000000001+0j)', span=1e-24, iterations=128))
        self.assertEqual(fallback_counts[0], 0)
        self.assertGreater(fallback_counts[1], 0)
        self.assertTrue((renderer.counts >= 0).all())

    def test_reference_cap_falls_back_without_out_of_bounds_access(self):
        renderer = ColorRenderer(1, 1)
        renderer.render(View(center=0j, span=1e-18, iterations=65537))
        self.assertEqual(len(renderer._reference_orbit), 65537)
        self.assertEqual(renderer.counts.item(), 65537)
        self.assertTrue((renderer.image == 0).all())
