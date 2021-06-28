import unittest
import numpy as np


import knee.curvature as curvature


class TestL_Method(unittest.TestCase):
    def test_get_knee(self):
        x = np.array([0,1,2,3,4,5,6,7,8,9,])
        y = np.array([1,0.5,0.333333333,0.25,0.2,0.166666667,0.142857143,0.125,0.111111111,0.1])
        points = np.stack((x, y), axis=1)
        result = curvature.knee(points)
        desired = 1
        self.assertEqual(result, desired)
    
    def test_multi_knee(self):
        x = np.array([0,1,2,3,4,5,6,7,8,9,])
        y = np.array([1,0.5,0.333333333,0.25,0.2,0.2,0.1,0.06666666667,0.05,0.04])
        points = np.stack((x, y), axis=1)
        result = curvature.multi_knee(points, t1=0.9)
        desired = np.array([1, 6])
        np.testing.assert_array_equal(result, desired)


if __name__ == '__main__':
    unittest.main()