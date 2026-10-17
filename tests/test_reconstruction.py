"""Checks for world-anchored sampling and tent reconstruction."""

import unittest
from unittest.mock import patch

import torch
import warp as wp

from mandelbrot import ColorRenderer, View, tent_resolve


class ReconstructionTests(unittest.TestCase):
    def test_tent_preserves_constant_and_linear_fields(self):
        # Linear fields should interpolate exactly at fractional camera offsets.
        y, x = torch.meshgrid(torch.arange(16, device="cuda"),
                              torch.arange(18, device="cuda"), indexing="ij")
        samples = torch.stack((x/18, y/16, torch.full_like(x, 1.0, dtype=torch.float32)), -1)
        image = torch.empty((3, 4, 3), device="cuda")
        stream = wp.stream_from_torch(torch.cuda.current_stream())
        wp.launch(tent_resolve, dim=(3, 4),
                  inputs=[wp.from_torch(samples), 2, 3.25, 4.125], outputs=[wp.from_torch(image)],
                  device="cuda", stream=stream)
        py, px = torch.meshgrid(torch.arange(3, device="cuda"),
                                torch.arange(4, device="cuda"), indexing="ij")
        expected = torch.stack(((3.25+(px+.5)*2-.5)/18,
                                (4.125+(py+.5)*2-.5)/16, torch.ones_like(px)), -1)
        torch.testing.assert_close(image, expected)

    def test_pan_and_recolor_reuse_orbits_and_return_to_same_image(self):
        renderer = ColorRenderer(24, 18, side=2, filter="tent")
        view = View(center=-.9284411855746013+.2626173711851386j, span=6.17e-10)
        initial = renderer.render(view).clone()
        grid = renderer._grid
        counts = grid.counts.clone()
        with patch.object(grid, "render", wraps=grid.render) as render:
            renderer.render(view.shifted(complex(view.span/200, view.span/300)))
            self.assertFalse(torch.equal(initial, renderer.image))
            renderer.render(view)
            torch.testing.assert_close(renderer.image, initial, rtol=0, atol=0)
            renderer.recolor(style="fire")
            self.assertEqual(render.call_count, 0)
        self.assertTrue(torch.equal(counts, grid.counts))
        self.assertTrue(torch.isfinite(renderer.image).all())

    def test_cache_refresh_is_integer_aligned_and_deep_pan_is_retained(self):
        view = View(center="(-0.743643887037151+0.13182590420533j)", span=1e-20)
        renderer = ColorRenderer(12, 10, side=2, filter="tent")
        renderer.render(view)
        grid = renderer._grid
        # Cross the overscan margin without changing zoom: reuse lattice anchor.
        moved = view.shifted(complex(view.span, view.span/20))
        renderer.render(moved)
        self.assertIs(renderer._grid, grid)
        self.assertEqual(renderer._grid_origin, (24, -1))
        self.assertEqual(renderer._grid_anchor.center_exact, view.center_exact)
        self.assertNotEqual(renderer.view.center_exact, view.center_exact)
        self.assertTrue(torch.isfinite(renderer.image).all())
        # Changing zoom must allocate a grid at the new density.
        renderer.render(moved.shifted(0j, span=view.span/2))
        self.assertIsNot(renderer._grid, grid)


if __name__ == "__main__":
    unittest.main()
