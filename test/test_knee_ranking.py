
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

    def test_rank_min_tol_matches_rank_min_when_values_are_separated(self):
        array = np.array([3.0, 1.0, 2.0, 1.0])
        np.testing.assert_array_equal(ranking.rank_min_tol(array), ranking.rank_min(array))

    def test_rank_min_tol_ties_values_differing_only_in_the_last_bits(self):
        # rank_min splits these into consecutive ranks because they are not
        # bit-identical; rank_min_tol must see one tied group.
        array = np.array([1.0, 1.0 + 3e-16, 1.0 - 2e-16, 2.0])
        np.testing.assert_array_equal(ranking.rank_min_tol(array), np.array([0, 0, 0, 3]))
        self.assertGreater(len(set(ranking.rank_min(array).tolist())), 2)

    def test_rank_min_tol_keeps_differences_above_the_tolerance(self):
        array = np.array([1.0, 1.1, 1.2])
        np.testing.assert_array_equal(ranking.rank_min_tol(array), np.array([0, 1, 2]))

    def test_rank_min_tol_atol_groups_values_around_zero(self):
        # Relative tolerance alone cannot tie 0.0 to a small non-zero value;
        # that is what atol is for.
        array = np.array([0.0, 1e-18, 1.0])
        np.testing.assert_array_equal(ranking.rank_min_tol(array), np.array([0, 1, 2]))
        np.testing.assert_array_equal(ranking.rank_min_tol(array, atol=1e-15), np.array([0, 0, 2]))

    def test_rank_min_tol_handles_empty_and_single_element(self):
        self.assertEqual(len(ranking.rank_min_tol(np.array([]))), 0)
        np.testing.assert_array_equal(ranking.rank_min_tol(np.array([5.0])), np.array([0]))


class TestArgmaxTol(unittest.TestCase):
    def test_exact_ties_take_the_lowest_index(self):
        self.assertEqual(ranking.argmax_tol(np.array([0.5, 0.5, 0.5])), 0)

    def test_last_bit_difference_is_still_a_tie(self):
        # np.argmax would return 1 here; that is the whole defect.
        values = np.array([0.5, 0.5 + 1e-16, 0.5])
        self.assertEqual(int(np.argmax(values)), 1)
        self.assertEqual(ranking.argmax_tol(values), 0)

    def test_real_differences_are_preserved(self):
        self.assertEqual(ranking.argmax_tol(np.array([0.5, 0.9, 0.5])), 1)
        self.assertEqual(ranking.argmax_tol(np.array([1.0, 1.1, 1.2])), 2)

    def test_matches_argmax_when_values_are_separated(self):
        values = np.array([1.0, 5.0, 3.0])
        self.assertEqual(ranking.argmax_tol(values), int(np.argmax(values)))

    def test_keys_resolve_the_tie(self):
        values = np.array([0.5, 0.5, 0.5])
        self.assertEqual(ranking.argmax_tol(values, keys=np.array([9, 2, 7])), 1)

    def test_works_on_negative_values(self):
        # The tolerance scales with |best|, so a negative maximum must still
        # tie correctly rather than widening or inverting the window.
        self.assertEqual(ranking.argmax_tol(np.array([-2.0, -1.0, -1.0 - 1e-16])), 1)

    def test_atol_covers_a_maximum_of_zero(self):
        # Relative tolerance alone cannot tie anything to 0.0.
        self.assertEqual(ranking.argmax_tol(np.array([0.0, -1e-18])), 0)
        self.assertEqual(ranking.argmax_tol(np.array([-1e-18, 0.0]), atol=1e-15), 0)

    def test_single_element_and_empty(self):
        self.assertEqual(ranking.argmax_tol(np.array([3.0])), 0)
        with self.assertRaises(ValueError):
            ranking.argmax_tol(np.array([]))


class TestSlopeRankingDeterminism(unittest.TestCase):
    """`slope_ranking` ranked its neighbourhood slopes with `rank`, which
    splits a tied group into consecutive integers ordered by whatever the
    sort produced. On a pure linear decline - where every neighbourhood
    slope is identical by construction - that turned last-bit noise into a
    winner, returning [0, 0.667, 0.333, 1] and picking the rightmost knee."""

    def setUp(self):
        self.costs = np.linspace(1.0, 0.05, 30)
        self.knees = np.array([5, 10, 15, 20])

    def _points(self, costs):
        return np.column_stack((np.arange(len(costs), dtype=float), costs))

    def test_equal_slopes_share_a_rank(self):
        scores = ranking.slope_ranking(self._points(self.costs), self.knees, t=0.8)
        self.assertEqual(len(set(scores.tolist())), 1)

    def test_choice_is_invariant_to_last_bit_noise(self):
        rng = np.random.default_rng(0)
        winners = set()
        for _ in range(100):
            jittered = self.costs + rng.uniform(-1e-15, 1e-15, self.costs.size)
            scores = ranking.slope_ranking(self._points(jittered), self.knees, t=0.8)
            winners.add(int(self.knees[ranking.argmax_tol(scores)]))
        self.assertEqual(winners, {5})

    def test_a_genuinely_steeper_knee_still_wins(self):
        # The tolerance must not flatten real differences into one group.
        costs = np.concatenate([np.linspace(1.0, 0.3, 8), np.full(22, 0.3)])
        scores = ranking.slope_ranking(self._points(costs), np.array([3, 12, 20]), t=0.8)
        self.assertGreater(len(set(scores.tolist())), 1)


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

    def test_choice_is_invariant_to_last_bit_noise(self):
        # The regression test for the real defect: on a linear decline every
        # knee's ratio is 1.0 to within ~1e-15, so ranking them exactly turns
        # last-bit arithmetic into a full integer rank spread - far more than
        # the 1e-6 leftmost nudge can overcome. The winner then depends on the
        # platform's libm rather than on the curve, which is how this first
        # showed up: the same input picked knee 2 on one machine and knee 20
        # on another. Perturbing below the noise floor must change nothing.
        costs = np.linspace(1.0, 0.05, 30)
        knees = np.array([2, 10, 20, 27])
        rng = np.random.default_rng(0)
        winners = set()
        for _ in range(100):
            jittered = costs + rng.uniform(-1e-15, 1e-15, costs.size)
            points = np.column_stack((np.arange(30, dtype=float), jittered))
            scores = ranking.right_flatness_ranking(points, knees, basis='left_ratio', flatness_weight=1.0)
            winners.add(int(knees[int(np.argmax(scores))]))
        self.assertEqual(winners, {2})

    def test_genuinely_flatter_remainder_still_beats_the_leftmost(self):
        # The tolerance must not be so eager that it merges real differences
        # and collapses the ranker into plain leftmost: knee 1 sits mid-decline
        # and knee 6 is deep in the flat tail, so 6 has to win over 8.
        costs = np.concatenate([np.linspace(1.0, 0.2, 6), np.full(14, 0.2)])
        points = np.column_stack((np.arange(20, dtype=float), costs))
        knees = np.array([1, 6, 8])
        scores = ranking.right_flatness_ranking(points, knees, basis='left_ratio', flatness_weight=1.0)
        self.assertEqual(knees[int(np.argmax(scores))], 6)

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