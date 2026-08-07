
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

import kneeliverse.knee_ranking as kr
import kneeliverse.postprocessing as pp
from kneeliverse import clustering


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


class TestTriangleArea(unittest.TestCase):
    def test_matches_the_shoelace_formula(self):
        # (0,1.0) (5,0.5) (10,0.3):
        #   0.5*|0*(0.5-0.3) + 5*(0.3-1.0) + 10*(1.0-0.5)| = 0.5*1.5 = 0.75
        p = np.array([[0.0, 1.0], [5.0, 0.5], [10.0, 0.3]])
        self.assertAlmostEqual(pp.triangle_area(p), 0.75)

    def test_collinear_points_enclose_nothing(self):
        p = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        self.assertAlmostEqual(pp.triangle_area(p), 0.0)

    def test_an_area_is_never_negative(self):
        # The shoelace determinant flips sign with the winding order; an area
        # cannot. This used to return -2.0 for one ordering and 2.0 for the
        # other, so a caller ranking corners by "area" would have ranked half
        # of them below zero.
        p = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]])
        self.assertAlmostEqual(pp.triangle_area(p), 2.0)
        self.assertGreaterEqual(pp.triangle_area(p), 0.0)

    def test_winding_order_does_not_change_the_area(self):
        p = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]])
        self.assertAlmostEqual(pp.triangle_area(p), pp.triangle_area(p[::-1]))

    def test_area_scales_with_the_height_of_the_corner(self):
        # Same base, taller apex -> larger area, monotonically.
        areas = [pp.triangle_area(np.array([[0.0, 0.0], [1.0, h], [2.0, 0.0]]))
                 for h in (0.5, 1.0, 2.0, 4.0)]
        self.assertEqual(areas, sorted(areas))

    def test_it_returns_a_plain_float(self):
        p = np.array([[0.0, 1.0], [5.0, 0.5], [10.0, 0.3]])
        self.assertIsInstance(pp.triangle_area(p), float)


class TestCornerRanking(unittest.TestCase):
    """Both rankers score how corner-like each knee is; they disagree on the
    measure, so only their shape and ordering behaviour is common."""

    def setUp(self):
        y = np.concatenate([np.linspace(1.0, 0.3, 8), np.full(12, 0.3)])
        self.points = np.column_stack((np.arange(len(y), dtype=float), y))
        self.knees = np.array([3, 7, 12])

    def test_one_score_per_knee(self):
        for fn in (pp.rank_corners, pp.rank_corners_triangle):
            with self.subTest(fn=fn.__name__):
                self.assertEqual(len(fn(self.points, self.knees)), len(self.knees))

    def test_scores_are_finite_and_non_negative(self):
        for fn in (pp.rank_corners, pp.rank_corners_triangle):
            with self.subTest(fn=fn.__name__):
                scores = fn(self.points, self.knees)
                self.assertTrue(np.all(np.isfinite(scores)))
                self.assertTrue(np.all(scores >= 0))

    def test_triangle_ranking_prefers_the_real_corner(self):
        # The curve bends at 7; knees 3 and 12 sit mid-run, where the three
        # points are collinear and enclose no area at all. Under the formula
        # this function used to inline - 0.5*(x1-x0)*(y1-y2), the left x-gap
        # times the RIGHT y-drop - knee 3 won instead, because a one-sided
        # step is not a measure of bending.
        scores = pp.rank_corners_triangle(self.points, np.array([3, 7, 12]))
        self.assertEqual(int(np.argmax(scores)), 1)
        self.assertAlmostEqual(scores[0], 0.0)
        self.assertAlmostEqual(scores[2], 0.0)
        self.assertGreater(scores[1], 0.0)

    def test_triangle_ranking_is_never_negative(self):
        # The old formula was signed, so a knee in a rising run scored below
        # zero and could never win its cluster however sharp it was. On
        # usr0.csv that happened to 1 of 9 real knees.
        rising = np.concatenate([np.linspace(0.3, 1.0, 8), np.full(12, 1.0)])
        points = np.column_stack((np.arange(len(rising), dtype=float), rising))
        scores = pp.rank_corners_triangle(points, np.array([3, 7, 12]))
        self.assertTrue(np.all(scores >= 0.0))

    def test_triangle_ranking_agrees_with_triangle_area(self):
        # It delegates now, rather than inlining a second, different formula.
        knees = np.array([3, 7, 12])
        expected = [pp.triangle_area(self.points[[k - 1, k, k + 1]]) for k in knees]
        np.testing.assert_allclose(pp.rank_corners_triangle(self.points, knees), expected)


class TestKneeFilters(unittest.TestCase):
    def setUp(self):
        y = np.concatenate([np.linspace(1.0, 0.3, 8), np.full(12, 0.3)])
        self.points = np.column_stack((np.arange(len(y), dtype=float), y))
        self.knees = np.array([3, 7, 12])

    def test_filters_return_a_subset_of_the_input(self):
        for fn in (pp.select_corner_knees, pp.filter_worst_knees):
            with self.subTest(fn=fn.__name__):
                out = fn(self.points, self.knees)
                self.assertTrue({int(i) for i in out} <= {int(i) for i in self.knees})

    def test_filters_preserve_order(self):
        out = pp.filter_worst_knees(self.points, self.knees)
        self.assertEqual(list(out), sorted(out))

    def test_filter_clusters_corners_returns_a_subset(self):
        out = pp.filter_clusters_corners(self.points, self.knees, clustering.average_linkage, t=0.05)
        self.assertTrue({int(i) for i in out} <= {int(i) for i in self.knees})

    def test_a_single_knee_survives_every_filter(self):
        single = np.array([7])
        for fn in (pp.filter_worst_knees,
                   lambda p, k: pp.filter_clusters_corners(p, k, clustering.average_linkage)):
            with self.subTest(fn=getattr(fn, '__name__', 'filter_clusters_corners')):
                self.assertEqual(len(fn(self.points, single)), 1)


class TestAddPointsEvenKnees(unittest.TestCase):
    """Fills gaps between knees so no pair is further apart than tx/ty."""

    def setUp(self):
        y = np.concatenate([np.linspace(1.0, 0.3, 8), np.full(12, 0.3)])
        self.points = np.column_stack((np.arange(len(y), dtype=float), y))
        self.knees = np.array([3, 7, 12])

    def test_it_only_adds(self):
        out = pp.add_points_even_knees(self.points, self.knees)
        self.assertGreaterEqual(len(out), len(self.knees))

    def test_the_result_is_sorted_and_unique(self):
        out = pp.add_points_even_knees(self.points, self.knees)
        self.assertEqual(list(out), sorted(set(out)))

    def test_every_index_is_within_the_curve(self):
        out = pp.add_points_even_knees(self.points, self.knees)
        self.assertTrue(np.all(out >= 0))
        self.assertTrue(np.all(out < len(self.points)))

    def test_a_tighter_spacing_adds_at_least_as_many(self):
        loose = pp.add_points_even_knees(self.points, self.knees, tx=0.5, ty=0.5)
        tight = pp.add_points_even_knees(self.points, self.knees, tx=0.02, ty=0.02)
        self.assertGreaterEqual(len(tight), len(loose))


class TestFilterClustersDeterminism(unittest.TestCase):
    """`filter_clusters` picks one knee per cluster by ranking the members
    and taking the best. It ranked with `rank` and then `np.argmax`, so a
    cluster whose knees score equally had its winner decided by last-bit
    noise rather than by the curve."""

    @staticmethod
    def _curve():
        # A staircase: several knees per step, equally good within a step.
        y = np.concatenate([np.linspace(1.0, 0.6, 7), np.full(4, 0.6),
                            np.linspace(0.6, 0.25, 7), np.full(4, 0.25)])
        return np.column_stack((np.arange(len(y), dtype=float), y))

    def test_selection_is_invariant_to_last_bit_noise(self):
        base = self._curve()
        knees = np.array([3, 5, 6, 13, 15, 16])
        rng = np.random.default_rng(0)
        results = set()
        for _ in range(100):
            points = base.copy()
            points[:, 1] += rng.uniform(-1e-15, 1e-15, len(points))
            filtered = pp.filter_clusters(points, knees, clustering.average_linkage,
                                          t=0.05, method=kr.ClusterRanking.left)
            results.add(tuple(int(i) for i in filtered))
        self.assertEqual(len(results), 1)


if __name__ == '__main__':
    unittest.main()