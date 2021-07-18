import unittest
import numpy as np
import knee.postprocessing as pp

import logging


class TestPostProcessing(unittest.TestCase):
    
    def test_rect_overlap(self):
        
        amin = np.array([2,1])
        amax = np.array([5,5])
        
        bmin = np.array([3, 2])
        bmax = np.array([5, 7])

        rv = pp.rect_overlap(amin, amax, bmin, bmax)

        desired = 0.375
        self.assertEqual(rv, desired)