import unittest
import torch
from viewer import escape_sobel
from mandelbrot import ColorRenderer,View


class EscapeGradientTests(unittest.TestCase):
    def test_constant_escape_time_has_no_false_mask_boundary(self):
        values=torch.full((16,16),20.)
        coverage=torch.ones_like(values);coverage[:,:8]=0;values[:,:8]=-1
        self.assertLess(escape_sobel(values,coverage).max().item(),1e-7)
        self.assertEqual(escape_sobel(values,torch.zeros_like(coverage)).max().item(),0)

    def test_ramp_matches_sobel_scale(self):
        ramp=torch.arange(16).float()[None].expand(16,-1)
        image=escape_sobel(ramp,torch.ones_like(ramp),scale=10.)
        expected=-torch.expm1(torch.tensor(-.1))
        torch.testing.assert_close(image[3:-3,3:-3],expected.expand(10,10,3))

    def test_palette_does_not_change_escape_gradient(self):
        renderer=ColorRenderer(16,16,side=2,filter='tent')
        renderer.render(View(center=-.75+.1j,span=.2,iterations=128))
        values,mask=renderer.escape_time_image();values=values.clone();mask=mask.clone()
        renderer.recolor(style='fire',phase=.6)
        other,coverage=renderer.escape_time_image()
        torch.testing.assert_close(other,values,rtol=0,atol=0)
        torch.testing.assert_close(coverage,mask,rtol=0,atol=0)
        self.assertTrue(torch.isfinite(escape_sobel(other,coverage)).all())

    def test_g_toggles_and_does_not_interrupt_text_entry(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from viewer import Viewer
        viewer=Viewer.__new__(Viewer)
        io=SimpleNamespace(want_text_input=False)
        viewer.imgui=SimpleNamespace(get_io=lambda:io)
        viewer.renderer=SimpleNamespace(escape_time_image=lambda:None)
        viewer.gradient_mode=False
        viewer.recolor_needed=False
        with patch.dict(sys.modules,{"pyglet.window":SimpleNamespace(key=SimpleNamespace(G=103))}):
            viewer.on_key_press(103,0)
            self.assertTrue(viewer.gradient_mode)
            self.assertTrue(viewer.recolor_needed)
            viewer.on_key_press(103,0)
            self.assertFalse(viewer.gradient_mode)
            io.want_text_input=True
            viewer.on_key_press(103,0)
            self.assertFalse(viewer.gradient_mode)

    def test_auto_contrast_reveals_weak_gradients_without_lighting_flat_fields(self):
        ramp=torch.arange(16).float()[None].expand(16,-1)*.01
        coverage=torch.ones_like(ramp)
        self.assertLess(escape_sobel(ramp,coverage,500).max().item(),.001)
        self.assertGreater(escape_sobel(ramp,coverage,None)[3:-3,3:-3].mean().item(),.5)
        self.assertEqual(escape_sobel(torch.zeros_like(ramp),coverage,None).max().item(),0)
