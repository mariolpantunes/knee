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
import kneeliverse.convex_hull as convex_hull


class TestConvexHull(unittest.TestCase):
    def test_ccw_colinear(self):
        a = np.array([-1,0])
        b = np.array([0,0])
        c = np.array([1,0])
        result = convex_hull._ccw(a,b,c)
        desired = 0
        self.assertEqual(result, desired)
    
    def test_ccw_ccw(self):
        a = np.array([1, 0])
        b = np.array([0, 1])
        c = np.array([-1,0])
        result = convex_hull._ccw(a,b,c)
        desired = 0
        self.assertGreater(result, desired)
    
    def test_ccw_cw(self):
        a = np.array([-1,0])
        b = np.array([0, 1])
        c = np.array([1, 0])
        result = convex_hull._ccw(a,b,c)
        desired = 0
        self.assertLess(result, desired)
    
    def test_sort_points(self):
        points = np.array([[0, 0], [0, 1], [0.3, 0.7], [0.7, 0.3], [1, 0]])
        result = convex_hull._sort_points(points)
        desired = np.array([[0, 0], [0, 1], [0.3, 0.7], [0.7, 0.3], [1, 0]])
        np.testing.assert_array_equal(result, desired)
    
    
    def test_graham_scan(self):
        points = np.array([[1,5], [1, 4], [2, 3], [2, 2], [2,4], [3,4], [4,3], [3,3], [5,1]])
        #points = np.array([[0, 0], [0, 1], [0.3, 0.7], [0.7, 0.3], [1, 0]])
        
        result = convex_hull.graham_scan(points)
        desired = np.array([1, 0, 5, 6, 8, 3])

        #import matplotlib.pyplot as plt
        #x = points[:, 0]
        #y = points[:, 1]
        #plt.scatter(x, y)
        #hull_points = points[result]
        #x = hull_points[:, 0]
        #y = hull_points[:, 1]
        #plt.plot(x, y, 'o', mec='r', color='none', lw=1, markersize=10)
        #plt.fill(x, y, edgecolor='r', fill=False)
        #plt.show()
        
        np.testing.assert_array_equal(result, desired)


if __name__ == '__main__':
    unittest.main()