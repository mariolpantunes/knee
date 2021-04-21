import unittest
import numpy as np


from knee.linear_fit import linear_fit_points, linear_r2_points

class TestLinearFir(unittest.TestCase):
    def test_r2_two(self):
        points = np.array([[0.0, 1.0], [1.0, 5.0]])
        coef = linear_fit_points(points)
        result = linear_r2_points(points, coef)
        desired = 1.0
        self.assertEqual(result, desired)