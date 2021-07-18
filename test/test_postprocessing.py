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
        result = pp.rect_overlap(amin, amax, bmin, bmax)
        desired = 0.375
        self.assertEqual(result, desired)
    
    def test_filter_corner_knees(self):
        points = np.array([[1,3], [2,3], [3,3], [4,2.5], [5,2], [6,1.5], [7, 1]])
        knees = np.array([1,2,3])
        result = pp.filter_corner_knees(points, knees, .5)
        desired = np.array([1,3])
        np.testing.assert_array_equal(result, desired)