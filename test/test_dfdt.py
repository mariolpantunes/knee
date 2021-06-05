import unittest
import numpy as np


import knee.dfdt as dfdt


class TestL_Method(unittest.TestCase):
    def test_get_knee_naive(self):
        x = np.array([0,1,2,3,4,5,6,7,8,9,])
        y = np.array([1,0.5,0.333333333,0.25,0.2,0.166666667,0.142857143,0.125,0.111111111,0.1])
        result = dfdt.get_knee(x,y)
        print(f'result {result}')
        desired = 2
        self.assertEqual(result, desired)

    def test_get_knee(self):
        x = np.array([0,1,2,3,4,5,6,7,8,9,])
        y = np.array([1,0.5,0.333333333,0.25,0.2,0.166666667,0.142857143,0.125,0.111111111,0.1])
        points = np.stack((x, y), axis=1)
        result = dfdt.knee_points(points)
        desired = 2
        self.assertEqual(result, desired)

