# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''

import math
import unittest
import numpy as np
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
    
    def test_rpd(self):
        coef = (0, 1)
        points = np.array([[0.0, 0.0], [1.0, 0.9], [2.0, 1.5], [3,2.25], [4,3.6], [5.0,5.0]])
        result = lf.rpd_points(points, coef)
        desired = 0.116
        self.assertAlmostEqual(result, desired, 2)
    
    def test_r2(self):
        coef = (0, 1)
        points = np.array([[0.0, 0.0], [1.0, 0.9], [2.0, 1.5], [3,2.25], [4,3.6], [5.0,5.0]])
        result = lf.r2_points(points, coef)
        desired = 0.973
        self.assertAlmostEqual(result, desired, 2)
    
    def test_rmspe(self):
        coef = (0, 1)
        points = np.array([[0.0, 0.0], [1.0, 0.9], [2.0, 1.5], [3,2.25], [4,3.6], [5.0,5.0]])
        result = lf.rmspe_points(points, coef)
        desired = 0.202
        self.assertAlmostEqual(result, desired, 2)


if __name__ == '__main__':
    unittest.main()