import math
import unittest
import numpy as np
import knee.knee_ranking as ranking


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
    
    def test_corner_ranking(self):
        points = np.array([[0,1],[1,1],[2,0],[3,0]])
        knees = np.array([1,2])
        result = ranking.corner_ranking(points, knees)
        desired = np.array([0, .5])
        np.testing.assert_array_equal(result, desired)
