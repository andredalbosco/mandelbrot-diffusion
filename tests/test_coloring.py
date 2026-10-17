"""Coloring must preserve geometry, deep coordinates, and cached interaction."""
import math
import time
import unittest
from decimal import Decimal, localcontext
from unittest.mock import patch

import torch
import warp as wp

from mandelbrot import ColorRenderer, View
from viewer import ProgressiveRenderer, Viewer


def orbit_reference(view, width, height):
    fields = torch.zeros(height, width, 2)
    with localcontext() as ctx:
        ctx.prec = 65
        step = Decimal(str(view.span))/width
        for y in range(height):
            for x in range(width):
                cx = Decimal(view.center_exact[0])+(Decimal(x)+Decimal('.5')-Decimal(width)/2)*step
                cy = Decimal(view.center_exact[1])+(Decimal(height)/2-y-Decimal('.5'))*step
                zx, zy = Decimal(str(view.z0.real)), Decimal(str(view.z0.imag))
                trap = Decimal('Infinity')
                n = 0
                while n < view.iterations and zx*zx+zy*zy <= 4:
                    zx, zy = zx*zx-zy*zy+cx, 2*zx*zy+cy
                    trap = min(trap, abs(zx), abs(zy))
                    n += 1
                if zx*zx+zy*zy <= 4:
                    continue
                if n == 0:
                    trap = min(abs(zx), abs(zy))
                z, c = complex(float(zx), float(zy)), complex(float(cx), float(cy))
                for _ in range(3):
                    if abs(z)**2 < 1e15:
                        z = z*z+c
                        n += 1
                smooth = n+1-math.log2(math.log(abs(z)))
                f = smooth % 1
                z /= abs(z)
                total = 0
                for k in range(5):
                    total += (1-f if k == 0 else 1)*z.imag
                    z *= z
                fields[y,x] = torch.tensor([.5+.5*(total+f*z.imag)/5, float(trap)])
    return fields


class ColoringTests(unittest.TestCase):
    def test_reset_selected_colors_preserves_view_and_all_reset_restores_defaults(self):
        viewer = Viewer.__new__(Viewer)
        viewer.renderer = ColorRenderer(8, 6, view=View(center=.3+.5j, span=.02, iterations=1024))
        viewer.renderer.set_colors(style='3b1b', lower=3.4, phase=.2, trap_scale=.5)
        view = viewer.renderer.view
        viewer.renderer.reset_colors()
        self.assertEqual(viewer.renderer.style, '3b1b')
        self.assertEqual(viewer.renderer.lower, 1.5)
        self.assertEqual(viewer.renderer.trap_scale, .5)
        self.assertEqual(viewer.renderer.view, view)
        viewer.gradient_mode = True
        viewer.gradient_scale = 7.
        viewer.reset_parameters()
        self.assertEqual(viewer.renderer.view, View())
        self.assertEqual(viewer.renderer.color_settings(), ColorRenderer.color_defaults)
        self.assertFalse(viewer.gradient_mode)
        self.assertTrue(viewer.render_needed)

    def test_orbit_fields_against_decimal_including_deep_coordinates_and_z0(self):
        views = [View(iterations=100), View(z0=.2+.1j, iterations=100),
                 View(center=0j, z0=3+1j, span=.1),
                 View(center='(-2.0000000000000000000000001+0j)', span=1e-24, iterations=128)]
        for view in views:
            with self.subTest(view=view):
                renderer = ColorRenderer(8, 6)
                renderer.render(view)
                actual = renderer.orbit_fields().cpu()
                expected = orbit_reference(view, 8, 6)
                torch.testing.assert_close(actual[...,0], expected[...,0], atol=2e-4, rtol=2e-4)
                torch.testing.assert_close(actual[...,1], expected[...,1], atol=1e-32, rtol=2e-4)

    def test_all_schemes_keep_escape_data_and_black_interior(self):
        renderer = ColorRenderer(16, 12, side=2)
        renderer.render()
        escape = renderer.escape.clone()
        counts = renderer.counts.clone()
        images = []
        for style in renderer.styles:
            images.append(renderer.recolor(style=style).clone())
            self.assertTrue(torch.isfinite(renderer.image).all())
            self.assertTrue(((renderer.image >= 0) & (renderer.image <= 1)).all())
            self.assertTrue((renderer.samples[escape == -1] == 0).all())
            torch.testing.assert_close(renderer.escape, escape, rtol=0, atol=0)
            torch.testing.assert_close(renderer.counts, counts, rtol=0, atol=0)
        for i in range(len(images)):
            for j in range(i):
                self.assertFalse(torch.equal(images[i], images[j]))

    def test_orbit_controls_and_pan_reuse_fields(self):
        renderer = ColorRenderer(16, 12, side=2, filter='tent')
        renderer.render()
        renderer.recolor(style='stripes')
        grid = renderer._grid
        from deep_zoom import orbit_fields_kernel
        with patch.object(grid, 'render', wraps=grid.render) as render, patch.object(wp, 'launch', wraps=wp.launch) as launch:
            renderer.recolor(stripe_contrast=2)
            renderer.recolor(style='orbit_trap', trap_scale=.5)
            renderer.render(renderer.view.shifted(.001j))
            self.assertEqual(render.call_count, 0)
            self.assertFalse(any(c.args[0] is orbit_fields_kernel for c in launch.call_args_list if c.args))
        self.assertTrue(torch.isfinite(renderer.image).all())

    def test_adaptation_uses_visible_samples_and_settles_without_new_orbits(self):
        renderer = ColorRenderer(16, 12, filter='tent')
        renderer.render()
        grid = renderer._grid
        grid.escape.fill_(1)
        grid.styled_escape.fill_(1e8)  # Overscan must not set the visible contrast.
        field = renderer._visible(grid.styled_escape)
        field.copy_(torch.linspace(4, 64, field.numel(), device='cuda').reshape_as(field))
        with patch('mandelbrot.time.monotonic', return_value=10.):
            renderer.recolor(style='3b1b_adaptive', adaptive_clip=0, adaptive_seconds=1)
        torch.testing.assert_close(renderer._adaptive_range, field.new_tensor([2, 8]))
        field.mul_(4)
        renderer._adaptive_target_key = None
        with patch('mandelbrot.time.monotonic', return_value=11.):
            renderer.recolor()
        expected = torch.lerp(field.new_tensor([2, 8]), field.new_tensor([4, 16]), 1-math.exp(-1))
        torch.testing.assert_close(renderer._adaptive_range, expected)
        self.assertTrue(renderer.adaptive_pending)
        with patch.object(grid, 'render', side_effect=AssertionError('replayed orbits')):
            with patch('mandelbrot.time.monotonic', return_value=30.):
                renderer.recolor()
        self.assertFalse(renderer.adaptive_pending)
        torch.testing.assert_close(renderer._adaptive_range, field.new_tensor([4, 16]))
        grid.escape.fill_(-1)
        renderer._adaptive_target_key = None
        renderer.recolor()
        self.assertTrue(torch.isfinite(renderer.image).all())
        self.assertEqual(renderer.image.count_nonzero().item(), 0)

    def test_progressive_adaptation_finishes_and_controls_round_trip(self):
        renderer = ColorRenderer(16, 12, side=2, filter='tent')
        renderer.set_colors(style='3b1b_adaptive', adaptive_seconds=.02, adaptive_clip=4,
                            smooth_scale=.4, histogram_strength=.8, stripe_contrast=2, trap_scale=.3)
        settings = renderer.color_settings()
        restored = ColorRenderer(16, 12, side=2, filter='tent')
        restored.set_colors(**settings)
        self.assertEqual(settings, restored.color_settings())
        engine = ProgressiveRenderer(renderer)
        try:
            for view in (View(iterations=128), View(center=-.75+.1j, span=.1, iterations=128)):
                engine.submit(view, settings)
                deadline = time.monotonic()+10
                last = None
                while time.monotonic() < deadline:
                    result = engine.poll()
                    if result:
                        last = result
                        self.assertTrue(result[3].isfinite().all())
                        if result[2] == 2 and not result[4]['adaptive_pending']:
                            break
                    time.sleep(.005)
                self.assertIsNotNone(last)
                self.assertEqual(last[2], 2)
                self.assertFalse(last[4]['adaptive_pending'])
        finally:
            engine.close()


if __name__ == '__main__':
    unittest.main()
