
__author__ = 'Mário Antunes'
__version__ = '1.0'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''

import unittest

import numpy as np

from kneeliverse import dfdt


class TestDFDT_Method(unittest.TestCase):
    def test_get_knee_naive(self):
        x = np.array([0,1,2,3,4,5,6,7,8,9,])
        y = np.array([1,0.5,0.333333333,0.25,0.2,0.166666667,0.142857143,0.125,0.111111111,0.1])
        result = dfdt.get_knee(x,y)
        desired = 1
        self.assertEqual(result, desired)

    def test_get_knee(self):
        x = np.array([0,1,2,3,4,5,6,7,8,9,])
        y = np.array([1,0.5,0.333333333,0.25,0.2,0.166666667,0.142857143,0.125,0.111111111,0.1])
        points = np.stack((x, y), axis=1)
        result = dfdt.knee(points)
        desired = 2
        self.assertEqual(result, desired)
    
    def test_multi_knee(self):
        x = np.array([0,1,2,3,4,5,6,7,8,9,])
        y = np.array([1,0.5,0.333333333,0.25,0.2,0.2,0.1,0.06666666667,0.05,0.04])
        points = np.stack((x, y), axis=1)
        result = dfdt.multi_knee(points, t1=0.5)
        desired = np.array([2])
        np.testing.assert_array_equal(result, desired)


class TestGetKneeGradient(unittest.TestCase):
    """`dfdt.knee` is a thin wrapper over this: it takes the gradient of the
    curve and returns where the rate of change settles."""

    def test_it_finds_the_corner(self):
        y = np.concatenate([np.linspace(1.0, 0.3, 8), np.full(12, 0.3)])
        self.assertEqual(dfdt.get_knee_gradient(np.gradient(y)), 7)

    def test_it_returns_a_plain_int(self):
        # It used to hand back a numpy integer, which is not an `int` as far
        # as a type checker (or a dict key) is concerned.
        y = np.concatenate([np.linspace(1.0, 0.3, 8), np.full(12, 0.3)])
        self.assertIsInstance(dfdt.get_knee_gradient(np.gradient(y)), int)

    def test_the_index_is_inside_the_curve(self):
        for corner in (4, 9, 14):
            with self.subTest(corner=corner):
                y = np.concatenate([np.linspace(1.0, 0.3, corner + 1),
                                    np.full(25 - corner - 1, 0.3)])
                idx = dfdt.get_knee_gradient(np.gradient(y))
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, len(y))


if __name__ == '__main__':
    unittest.main()