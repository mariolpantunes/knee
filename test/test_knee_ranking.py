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

import math
import unittest
import numpy as np
import kneeliverse.knee_ranking as ranking


class TestKneeRanking(unittest.TestCase):
    def test_rect_overlap(self):
        amin = np.array([2, 1])
        amax = np.array([5, 5])
        bmin = np.array([3, 2])
        bmax = np.array([5, 7])
        result = ranking.rect_overlap(amin, amax, bmin, bmax)
        desired = 0.375
        self.assertEqual(result, desired)
    
    def test_upper_overlap_00(self):
        points = np.array([[0,1],[1,1],[2,0],[3,0]])
        idx = 1
        p0, p1, p2 = points[idx-1:idx+2]
        corner0 = np.array([p0[0], p2[1]])
        amin, amax = ranking.rect(corner0, p1)
        bmin, bmax = ranking.rect(p0, p2)
        result = ranking.rect_overlap(amin, amax, bmin, bmax)
        desired = 0.5
        self.assertEqual(result, desired)
    
    def test_upper_overlap_01(self):
        points = np.array([[0,1],[1,1],[2,0],[3,0]])
        idx = 2
        p0, p1, p2 = points[idx-1:idx+2]
        corner0 = np.array([p0[0], p2[1]])
        amin, amax = ranking.rect(corner0, p1)
        bmin, bmax = ranking.rect(p0, p2)
        result = ranking.rect_overlap(amin, amax, bmin, bmax)
        desired = 0.0
        self.assertEqual(result, desired)
    
    def test_lower_overlap_00(self):
        points = np.array([[0,1],[1,1],[2,0],[3,0]])
        idx = 1
        p0, p1, p2 = points[idx-1:idx+2]
        corner0 = np.array([p2[0], p0[1]])
        amin, amax = ranking.rect(corner0, p1)
        bmin, bmax = ranking.rect(p0, p2)
        result = ranking.rect_overlap(amin, amax, bmin, bmax)
        desired = 0.0
        self.assertEqual(result, desired)
    
    def test_lower_overlap_01(self):
        points = np.array([[0,1],[1,1],[2,0],[3,0]])
        idx = 2
        p0, p1, p2 = points[idx-1:idx+2]
        corner0 = np.array([p2[0], p0[1]])
        amin, amax = ranking.rect(corner0, p1)
        bmin, bmax = ranking.rect(p0, p2)
        result = ranking.rect_overlap(amin, amax, bmin, bmax)
        desired = 0.5
        self.assertEqual(result, desired)
    
    def test_lower_overlap_02(self):
        points = np.array([[0,1],[1,0],[2,1]])
        idx = 1
        p0, p1, p2 = points[idx-1:idx+2]
        corner0 = np.array([p2[0], p0[1]])
        amin, amax = ranking.rect(corner0, p1)
        bmin, bmax = ranking.rect(p0, p2)
        result = ranking.rect_overlap(amin, amax, bmin, bmax)
        desired = 0.0
        self.assertEqual(result, desired)

    def test_lower_overlap_03(self):
        amin, amax = (np.array([782, 3.47059833e-01]), np.array([805, 3.49430416e-01]))
        bmin, bmax = (np.array([759, 3.48911963e-01]), np.array([805, 3.49430416e-01]))
        result = ranking.rect_overlap(amin, amax, bmin, bmax)
        desired = 0.17945536158082412
        self.assertEqual(result, desired)

    def test_distances(self):
        points = np.array([[0,1],[1,1],[2,0],[3,0]])
        point = np.array([0,0])
        result = ranking.distances(point, points)
        desired = np.array([1, math.sqrt(2.0), 2, 3])
        np.testing.assert_array_equal(result, desired)


if __name__ == '__main__':
    unittest.main()