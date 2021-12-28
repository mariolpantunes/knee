import unittest
import numpy as np

import math

import knee.linear_fit as lf

class TestLinearFir(unittest.TestCase):
    def test_r2_two(self):
        points = np.array([[0.0, 1.0], [1.0, 5.0]])
        coef = lf.linear_fit_points(points)
        result = lf.linear_r2_points(points, coef)
        desired = 1.0
        self.assertEqual(result, desired)
    
    def test_rmspe(self):
        x = np.array([0,1,2,3,4])
        y = np.array([2,2,2,2,2])
        coef = (2,0)
        result = lf.rmspe(x,y, coef)
        desired = 0.0
        self.assertEqual(result, desired)
    
    def test_angle_00(self):
        coef1 = (0, 0)
        result = lf.angle(coef1, coef1)
        desired = 0.0
        self.assertEqual(result, desired)
    
    def test_angle_01(self):
        coef1 = (0, 0)
        coef2 = (0, 1e10)
        result = math.degrees(math.fabs(lf.angle(coef1, coef2)))
        desired = 89.99999999427042
        self.assertAlmostEqual(result, desired, 2)
        