"""Numerical checks for the experimental batched generation renderer."""

from decimal import Decimal, localcontext
import unittest

import torch

from mandelbrot import ColorRenderer, View
from mandelbrot_fast import FastRenderer


def precise_counts(view, width, height, side):
    result = torch.empty((height*side+2*side, width*side+2*side), dtype=torch.int32)
    with localcontext() as context:
        context.prec = 70
        step = Decimal(str(view.span))/Decimal(width*side)
        for y in range(result.shape[0]):
            for x in range(result.shape[1]):
                cx = Decimal(view.center_exact[0])+(Decimal(x)+Decimal('.5')-Decimal(result.shape[1])/2)*step
                cy = Decimal(view.center_exact[1])+(Decimal(result.shape[0])/2-Decimal(y)-Decimal('.5'))*step
                zx, zy = Decimal(0), Decimal(0)
                n = 0
                while n < view.iterations and zx*zx+zy*zy <= 4:
                    zx, zy = zx*zx-zy*zy+cx, 2*zx*zy+cy
                    n += 1
                result[y, x] = n
    return result


class FastRendererTests(unittest.TestCase):
    def test_orbit_coloring_matches_independent_complex_iteration(self):
        import math
        views=[View(center=3+.2j,span=1.,iterations=80),View(center=0j,span=.1,iterations=80)]
        renderer=FastRenderer(views,width=8,height=8,side=2,precision='float64')
        original=renderer.render().clone()
        fields=renderer.orbit_coloring().cpu()
        escape=renderer.escape.cpu()
        self.assertTrue(torch.isfinite(escape).all())
        self.assertTrue(torch.isfinite(fields).all())
        self.assertTrue((fields[1]==0).all())
        h,w=renderer.counts.shape[-2:]
        for y,x in [(0,0),(h//2,w//2),(h-1,w-1)]:
            c=views[0].center+complex(x+.5-w/2,h/2-y-.5)*views[0].span/16
            z=0j; trap=float('inf')
            for _ in range(int(renderer.counts[0,y,x])):
                z=z*z+c;trap=min(trap,abs(z.real),abs(z.imag))
            for _ in range(3):
                if abs(z)**2<1e15:z=z*z+c
            z/=abs(z)
            f=float(escape[0,y,x])%1; total=0.
            for octave in range(5):
                total+=(1-f if octave==0 else 1)*z.imag
                z=z*z
            stripe=.5+.5*(total+f*z.imag)/5
            self.assertAlmostEqual(float(fields[0,y,x,0]),stripe,places=5)
            self.assertAlmostEqual(float(fields[0,y,x,1]),trap,places=5)
        palette=torch.tensor([[0.,.1,.8],[1.,.7,.1],[.2,.8,.3]],device='cuda')
        output=renderer.recolor(palette,fields[...,0].to('cuda')).clone()
        self.assertTrue((output[1]==0).all())
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater((output[0]-original[0]).abs().mean().item(),.01)

    def test_direct_batch_matches_scalar_renderer(self):
        views = [View(iterations=80), View(center=.25+.25j, span=2., iterations=80)]
        for precision in ('float64', 'hybrid32'):
            fast = FastRenderer(views, width=16, height=12, side=2, precision=precision)
            fast.render()
            for b, view in enumerate(views):
                reference = ColorRenderer(36, 28)
                reference.render(view.shifted(0j, span=view.span*36/32))
                self.assertTrue(torch.equal(fast.counts[b], reference.counts))
                torch.testing.assert_close(fast.samples[b], reference.image, atol=2e-5, rtol=2e-5)
            self.assertFalse(torch.equal(fast.image[0], fast.image[1]))

    def test_fp32_preview_in_known_interior_and_exterior_regions(self):
        # Boundary quality is measured separately, not promised for raw FP32.
        views=[View(center=0j,span=.2),View(center=3+0j,span=.25),View(center=1000+0j,span=10.)]
        fast=FastRenderer(views,width=16,height=12,precision='float32')
        fast.render()
        self.assertTrue((fast.image[0]==0).all())
        self.assertTrue((fast.counts[1]==1).all())
        self.assertTrue((fast.image[1]>0).all())
        self.assertTrue((fast.counts[2]==1).all())
        self.assertTrue((fast.image[2]>0).all())

    def test_deep_batch_matches_70_digit_counts(self):
        views = [
            View(center='(-2.0000000000000000000000001+0j)', span=1e-24, iterations=128),
            View(center='(-0.92844118559550850366513868844580424143709174+0.262617371450439604741270010095574502976117479j)', span=4.118046071574424e-20),
        ]
        expected = torch.stack([precise_counts(v, 8, 6, 2) for v in views])
        for precision in ('perturb32', 'perturb64'):
            for guard in ('strict', 'bailout'):
                with self.subTest(precision=precision, guard=guard):
                    fast = FastRenderer(views, width=8, height=6, side=2, precision=precision, tile=8, guard=guard)
                    fast.render()
                    self.assertTrue(torch.equal(fast.counts.cpu(), expected))
                    self.assertGreater(fast.fallbacks.numpy().sum(), 0)
                    self.assertGreater(fast.image[1].std().item(), .05)
                    self.assertTrue(torch.isfinite(fast.image).all())

    def test_fallback_queue_resets_on_a_torch_stream(self):
        fast=FastRenderer([View()],width=16,height=12,precision='hybrid32')
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream):
            fast.render()
            first=fast.counts.clone()
            fast.render()
        stream.synchronize()
        self.assertTrue(torch.equal(first,fast.counts))
        self.assertLessEqual(int(fast.fallbacks.numpy().sum()),fast.counts.numel())

    def test_precision_limits_are_explicit(self):
        for precision in ('float32','float64','perturb32','perturb64'):
            with self.assertRaises(ValueError):
                FastRenderer([View(span=1e-40)], precision=precision)
            with self.assertRaises(ValueError):
                FastRenderer([View(z0=1j)], precision=precision)
        with self.assertRaises(ValueError):
            FastRenderer([View(span=1e-8)], precision='float32')


class DatasetRendererTests(unittest.TestCase):
    def test_mixed_scales_keep_order_and_reference_pixels(self):
        from mandelbrot_fast import DatasetRenderer
        views = [View(center=0j, span=.1, iterations=64),
                 View(center=3+0j, span=3.5, iterations=64)]
        backend = DatasetRenderer(mode="float64")
        backend.calibrate(views)
        images, modes = backend.render(views)
        reference = FastRenderer(views, side=2, precision="float64").render().cpu()
        torch.testing.assert_close(images, reference, rtol=0, atol=0)
        self.assertEqual(modes, ["float64", "float64"])
        self.assertFalse(torch.equal(images[0], images[1]))

    def test_failed_production_check_recomputes_batch(self):
        from mandelbrot_fast import DatasetRenderer
        views = [View(center=-.75+.1j, span=.05, iterations=128)]
        backend = DatasetRenderer()
        backend.selected = {backend.bucket(views[0]): "hybrid32"}
        backend.report["rgb_mae_limit"] = -1  # Force rejection to exercise safe fallback.
        images, modes = backend.render(views)
        reference = FastRenderer(views, side=2, precision="float64").render().cpu()
        torch.testing.assert_close(images, reference, rtol=0, atol=0)
        self.assertEqual(modes, ["float64"])
        self.assertFalse(backend.report["checks"][0]["accepted"])


class ViewerSampleBufferTests(unittest.TestCase):
    def test_no_halo_escape_buffers_match_independent_counts(self):
        view=View(center='(-2.0000000000000000000000001+0j)',span=1e-24,iterations=128)
        fast=FastRenderer([view],width=8,height=6,side=1,precision='perturb32',guard='bailout',halo=False)
        fast.render(resolve=False)
        expected=precise_counts(view.shifted(0j,span=view.span*6/8),6,4,1)
        self.assertTrue(torch.equal(fast.counts[0].cpu(),expected))
        self.assertTrue(torch.isfinite(fast.escape).all())
        self.assertTrue(torch.isfinite(fast.styled_escape).all())
        self.assertTrue(torch.equal(fast.escape==-1,fast.counts==view.iterations))
        with self.assertRaises(ValueError):fast.render()

    def test_viewer_resolves_fast_samples_and_retains_escape_fields(self):
        from viewer import escape_sobel
        view=View(center='(-0.92844118559550850366513868844580424143709174+0.262617371450439604741270010095574502976117479j)',span=4.118046071574424e-20,iterations=512)
        renderer=ColorRenderer(16,12,side=2,filter='tent')
        renderer.render(view)
        self.assertIn('Tiled',renderer.precision)
        grid=renderer._grid
        self.assertTrue(torch.equal(grid.counts,grid._tiled_backend.counts[0]))
        self.assertTrue(torch.equal(grid.escape,grid._tiled_backend.escape[0]))
        field,coverage=renderer.escape_time_image()
        gradient=escape_sobel(field,coverage,None)
        self.assertTrue(torch.isfinite(gradient).all())
        renderer.recolor(style='3b1b')
        self.assertTrue(torch.isfinite(renderer.image).all())
        field2,coverage2=renderer.escape_time_image()
        torch.testing.assert_close(field2,field,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
