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

    def test_rank_min_ties_share_lowest_rank(self):
        array = np.array([3.0, 1.0, 1.0, 2.0, 1.0])
        result = ranking.rank_min(array)
        desired = np.array([4, 0, 0, 3, 0])
        np.testing.assert_array_equal(result, desired)

    def test_rank_min_no_ties_matches_rank(self):
        array = np.array([5.0, 2.0, 8.0, 1.0])
        np.testing.assert_array_equal(ranking.rank_min(array), ranking.rank(array))


class TestRightFlatnessRanking(unittest.TestCase):
    def test_prefers_earliest_knee_whose_remainder_is_flat(self):
        # A sharp early knee at k=3, then an exactly flat tail from k=3 onward.
        costs = np.concatenate([np.linspace(1.0, 0.2, 4), np.full(16, 0.2)])
        points = np.column_stack((np.arange(20, dtype=float), costs))
        knees = np.array([1, 3, 6, 10, 15])
        scores = ranking.right_flatness_ranking(points, knees, basis='left_ratio', flatness_weight=1.0)
        self.assertEqual(knees[int(np.argmax(scores))], 3)

    def test_does_not_falsely_flatten_a_still_declining_curve(self):
        # Pure, uniform linear decline - no knee's remainder is genuinely
        # flatter than any other's, so ties must resolve to the
        # deterministic leftmost fallback.
        costs = np.linspace(1.0, 0.05, 30)
        points = np.column_stack((np.arange(30, dtype=float), costs))
        knees = np.array([2, 10, 20, 27])
        scores = ranking.right_flatness_ranking(points, knees, basis='left_ratio', flatness_weight=1.0)
        self.assertEqual(knees[int(np.argmax(scores))], 2)

    def test_pure_leftmost_at_zero_weight(self):
        costs = np.concatenate([np.linspace(1.0, 0.2, 4), np.full(16, 0.2)])
        points = np.column_stack((np.arange(20, dtype=float), costs))
        knees = np.array([3, 6, 10, 15])
        scores = ranking.right_flatness_ranking(points, knees, flatness_weight=0.0)
        self.assertEqual(knees[int(np.argmax(scores))], 3)

    def test_overall_ratio_basis_runs_without_error(self):
        costs = np.concatenate([np.linspace(1.0, 0.2, 4), np.full(16, 0.2)])
        points = np.column_stack((np.arange(20, dtype=float), costs))
        knees = np.array([1, 3, 6, 10, 15])
        scores = ranking.right_flatness_ranking(points, knees, basis='overall_ratio', flatness_weight=0.5)
        self.assertEqual(len(scores), len(knees))


if __name__ == '__main__':
    unittest.main()