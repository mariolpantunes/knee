# coding: utf-8

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
import kneeliverse.rdp as rdp
import kneeliverse.postprocessing as pp


class TestPostProcessing(unittest.TestCase):
    
    def test_filter_corner_knees_00(self):
        points = np.array([[1,3], [2,3], [3,3], [4,2.5], [5,2], [6,1.5], [7, 1]])
        knees = np.array([1,2,3])
        result = pp.filter_corner_knees(points, knees, .5)
        desired = np.array([1,3])
        np.testing.assert_array_equal(result, desired)
    
    def test_filter_corner_knees_01(self):
        points = np.array([[33.0, 0.25715391], [4.29000000e+02, 2.49621243e-01], [4.62000000e+02, 1.72661497e-01]])
        knees = np.array([1])
        result = pp.filter_corner_knees(points, knees, .5)
        desired = np.array([])
        np.testing.assert_array_equal(result, desired)
    
    """def test_add_even_points_0(self):
        points = np.array([[0, 6], [1, 5], [2, 4], [3, 1], [4, 2], [5, 2], [6, 2], [7, 3], [8, 3], [9, 2], [10, 1], [11, 1/4], [12, 0]])
        knees = np.array([1,2])

        reduced, removed = rdp.rdp(points)

        print(f"{reduced} {removed}")

        result = pp.add_points_even(points, reduced, knees, removed,  extremes=False)
        desired = np.array([1,2,3,10,11,12])
        np.testing.assert_array_equal(result, desired)
    
    def test_add_even_points_1(self):
        points = np.array([[0, 6], [1, 5], [2, 4], [3, 1], [4, 2], [5, 2], [6, 2], [7, 3], [8, 3], [9, 2], [10, 1], [11, 1/4], [12, 0]])
        knees = np.array([1,2])

        points_reduced, removed = rdp.rdp(points)

        result = pp.add_points_even(points, points_reduced, knees, removed,  extremes=True)
        desired = np.array([0,1,2,3,10,11,12])
        np.testing.assert_array_equal(result, desired)"""


if __name__ == '__main__':
    unittest.main()