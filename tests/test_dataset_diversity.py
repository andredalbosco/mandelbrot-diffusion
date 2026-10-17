import unittest
from dataset import ViewDiversity, zoom_depth_quotas

class DiversityTests(unittest.TestCase):
    def test_quotas(self):
        quotas=zoom_depth_quotas(9000,12,1)
        self.assertEqual(sum(quotas),9000)
        self.assertEqual(quotas[0],1)
        self.assertLessEqual(max(quotas[1:])-min(quotas[1:]),1)
        self.assertEqual(zoom_depth_quotas(10,2),[4,3,3])

    def test_neighboring_cells_and_scales(self):
        index=ViewDiversity(.01)
        self.assertTrue(index.add((-.5001,.2001),1.00001))
        self.assertFalse(index.add((-.4999,.1999),.99999))
        self.assertFalse(index.add((-.5001,.2001),1.00001))
        self.assertTrue(index.add((-.45,.2),1.00001))
        self.assertTrue(index.add((-.5,.2),1.1))
