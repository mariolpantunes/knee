import unittest
import numpy as np

from knee.postprocessing import corner_point


import logging
logging.basicConfig(level=logging.INFO)

class TestPostProcessing(unittest.TestCase):
    def test_corner_point_true(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 4.0], [4.0, 3.0],
                           [5.0, 2.0], [6.0, 1.0], [7.0, 1.0], [8.0, 1.0]])
        result = corner_point(points, 1, 0.1)
        desired = True
        self.assertEqual(result, desired)
    
    def test_corner_point_false(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 4.0], [4.0, 3.0],
                           [5.0, 2.0], [6.0, 1.0], [7.0, 1.0], [8.0, 1.0]])
        result = corner_point(points, 5, 0.2)
        desired = False
        self.assertEqual(result, desired)