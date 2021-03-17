import unittest
import numpy as np


from knee.lmethod import get_knee, multiknee, multiknee_cache

import logging
logging.basicConfig(level=logging.DEBUG)

class TestL_Method(unittest.TestCase):
    def test_get_knee_naive(self):
        x = np.array([0,1,2,3,4,5,6,7,8,9,])
        y = np.array([1,0.5,0.333333333,0.25,0.2,0.166666667,0.142857143,0.125,0.111111111,0.1])
        result, coef_left, coef_right = get_knee(x, y)
        desired = 2
        self.assertEqual(result, desired)

    def test_multi_knee_dual(self):
        x = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,])
        y = np.array([1,1,0.7,0.538461538,0.4375,0.368421053,0.318181818,0.28,0.25,0.225806452,0.01,0.008264463,0.006944444,0.00591716,0.005102041,0.004444444,0.00390625,0.003460208,0.00308642,0.002770083])
        points = np.transpose(np.stack((x,y)))
        result = multiknee_cache(points, t = 0.99, debug=True)
        desired = multiknee(points, t= 0.99, debug=True)
        print(result)
        print(desired)
        self.assertEqual(result, desired)
