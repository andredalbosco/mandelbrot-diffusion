"""Check the Sobel score used to inspect and rank dataset views."""
import unittest
import torch
from dataset_viewer import histogram_coordinates, sobel_gradients


class SobelTests(unittest.TestCase):
    def test_flat_images_have_no_artificial_frame(self):
        gradient = sobel_gradients(torch.full((2,3,128,128),.7))
        self.assertLess(gradient.max().item(),1e-7)

    def test_linear_ramp_has_expected_gradient_and_rotation(self):
        ramp = torch.linspace(0,1,128)[None,None,None].expand(1,1,128,128)
        horizontal = sobel_gradients(ramp)
        vertical = sobel_gradients(ramp.transpose(-1,-2))
        torch.testing.assert_close(horizontal[:,:,3:-3,3:-3],torch.full((1,1,122,122),1/127),atol=1e-7,rtol=1e-5)
        torch.testing.assert_close(horizontal.transpose(-1,-2),vertical,atol=1e-7,rtol=1e-5)

    def test_step_scores_above_flat_and_blur_suppresses_checkerboard(self):
        step=torch.zeros(1,1,128,128);step[:,:,:,64:]=1
        checker=((torch.arange(128)[:,None]+torch.arange(128)[None,:])%2).float()[None,None]
        self.assertGreater(sobel_gradients(step).mean().item(),.005)
        self.assertLess(sobel_gradients(checker).max().item(),1e-7)

    def test_grayscale_and_rgb_agree(self):
        image=torch.rand(2,1,64,64)
        torch.testing.assert_close(sobel_gradients(image),sobel_gradients(image.expand(-1,3,-1,-1)),atol=1e-7,rtol=1e-5)


class HistogramColoringTests(unittest.TestCase):
    def test_pair_shares_mapping_without_other_pairs_affecting_it(self):
        parent = torch.tensor([[[-1.,2.,3.,4.]], [[-1.,20.,30.,40.]]])
        target = torch.tensor([[[2.,3.,4.,8.]], [[20.,30.,40.,80.]]])
        fields = torch.cat((parent,target))
        alpha = histogram_coordinates(fields)
        torch.testing.assert_close(alpha[0,0,1:],alpha[2,0,:3])
        changed = fields.clone(); changed[[1,3]] *= 10
        torch.testing.assert_close(alpha[[0,2]],histogram_coordinates(changed)[[0,2]])
        self.assertTrue((alpha[2,0,1:]>=alpha[2,0,:-1]).all())

    def test_empty_and_constant_fields_are_finite(self):
        for value in (-1., 10.):
            alpha = histogram_coordinates(torch.full((2,8,8),value))
            self.assertTrue(torch.isfinite(alpha).all())
            self.assertTrue(((alpha>=0)&(alpha<=1)).all())


if __name__=='__main__':
    unittest.main()
