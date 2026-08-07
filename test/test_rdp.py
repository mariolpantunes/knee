
__author__ = 'Mário Antunes'
__version__ = '1.0'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''

import os
import tempfile
import unittest

import numpy as np

from kneeliverse import linear_fit as lf
from kneeliverse import metrics, rdp


class TestRDP(unittest.TestCase):
    
    def test_rdp_00(self):
        points = np.array([[1, 5], [2, 5], [3, 5], [4, 5], [5, 5]])
        reduced, removed = rdp.rdp(points)
        desired = np.array([0, 4])
        np.testing.assert_array_equal(reduced, desired)
        desired = np.array([[0, 3]])
        np.testing.assert_array_equal(removed, desired)
    
    def test_rdp_01(self):
        points = np.array([[1, 5], [2, 5], [3, 6], [4, 6], [5, 6]])
        reduced, removed = rdp.rdp(points)
        desired = np.array([0, 1, 2, 4])
        np.testing.assert_array_equal(reduced, desired)
        desired = np.array([[0, 0],[1, 0],[2, 1]])
        np.testing.assert_array_equal(removed, desired)

    def test_rdp_mapping_line(self):
        points = np.array([[1, 5], [2, 5], [3, 5], [4, 5], [5, 5]])
        reduced, removed = rdp.rdp(points)
        indexes = np.array([0, 1])
        result = rdp.mapping(indexes, reduced, removed)
        desired = np.array([0, 4])
        np.testing.assert_array_equal(result, desired)
    
    def test_rdp_mapping_two(self):
        points = np.array([[0, 3], [1, 3], [2, 3], [3, 2], [4, 1], [5, 0]])
        reduced, removed = rdp.rdp(points)
        indexes = np.array([0, 1, 2])
        result = rdp.mapping(indexes, reduced, removed)
        desired = np.array([0, 2, 5])
        np.testing.assert_array_equal(result, desired)
    
    def test_rdp_mapping_four(self):
        points = np.array([[2, 0], [3, 1], [4, 2], [5, 2], [6, 2], [7, 3], [8, 4], [9, 3], [10, 2], [11, 1], [12, 0]])
        reduced, removed = rdp.rdp(points)
        indexes = np.array([0, 1, 2, 3, 4])
        result = rdp.mapping(indexes, reduced, removed)
        desired = np.array([0, 2, 4, 6, 10])
        np.testing.assert_array_equal(result, desired)
    
    def test_compute_removed_points_00(self):
        points = np.array([[1, 5], [2, 5], [3, 5], [4, 5], [5, 5]])
        reduced = np.array([0, 4])
        result = rdp.compute_removed_points(points, reduced)
        desired = np.array([[0,3]])
        np.testing.assert_array_equal(result, desired)
    
    def test_compute_removed_points_01(self):
        points = np.array([[1, 5], [2, 5], [3, 6], [4, 6], [5, 6]])
        reduced = np.array([0, 1, 2, 4])
        result = rdp.compute_removed_points(points, reduced)
        desired = np.array([[0, 0],[1, 0],[2, 1]])
        np.testing.assert_array_equal(result, desired)

    def test_grdp_00(self):
        points = np.array([[1, 5], [2, 5], [3, 5], [4, 5], [5, 5]])
        reduced, removed = rdp.grdp(points)
        desired = np.array([0, 4])
        np.testing.assert_array_equal(reduced, desired)
        desired = np.array([[0,3]])
        np.testing.assert_array_equal(removed, desired)
    
    def test_grdp_01(self):
        points = np.array([[1, 5], [2, 5], [3, 6], [4, 6], [5, 6]])
        reduced, removed = rdp.grdp(points)
        desired = np.array([0, 1, 2, 4])
        np.testing.assert_array_equal(reduced, desired)
        desired = np.array([[0,0],[1, 0],[2,1]])
        np.testing.assert_array_equal(removed, desired)

    def test_grdp_02(self):
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 2], [4, 3], [5, 4]])
        reduced, removed = rdp.grdp(points)
        desired = np.array([0, 2, 3, 5])
        np.testing.assert_array_equal(reduced, desired)
        desired = np.array([[0, 1], [2, 0], [3, 1]])
        np.testing.assert_array_equal(removed, desired)
    
    def test_fixed_rdp_00(self):
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 2], [4, 3], [5, 4]])
        reduced, removed = rdp.rdp_fixed(points, 3)
        desired = np.array([0, 2, 5])
        np.testing.assert_array_equal(reduced, desired)
        desired = np.array([[0, 1], [2, 2]])
        np.testing.assert_array_equal(removed, desired)

    def test_fixed_rdp_01(self):
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 2], [4, 3], [5, 4]])
        reduced, removed = rdp.rdp_fixed(points, 4)
        desired = np.array([0, 2, 3, 5])
        np.testing.assert_array_equal(reduced, desired)
        desired = np.array([[0, 1], [2, 0], [3, 1]])
        np.testing.assert_array_equal(removed, desired)
    
    def test_mp_grdp_00(self):
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 2], [4, 3], [5, 4]])
        reduced, removed = rdp.mp_grdp(points, min_points=3)
        desired = np.array([0, 2, 3, 5])
        np.testing.assert_array_equal(reduced, desired)
        desired = np.array([[0, 1], [2, 0], [3, 1]])
        np.testing.assert_array_equal(removed, desired)
    
    def test_mp_grdp_01(self):
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 2], [4, 3], [5, 4]])
        reduced, removed = rdp.mp_grdp(points, min_points=10)
        desired = np.array([0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(reduced, desired)
        desired = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
        np.testing.assert_array_equal(removed, desired)
    
    def test_mp_grdp_02(self):
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 2], [4, 3], [5, 4]])
        reduced, removed = rdp.mp_grdp(points, t = 0.1, min_points=4)
        desired = np.array([0, 2, 3, 5])
        np.testing.assert_array_equal(reduced, desired)
        desired = np.array([[0, 1], [2, 0], [3, 1]])
        np.testing.assert_array_equal(removed, desired)


class TestRDPDeterminism(unittest.TestCase):
    """RDP splits at the point furthest from the chord, and that distance is
    computed, not given. On a symmetric curve two points are equidistant to
    the last bit, and an exact `np.argmax` then picks between them on the
    arithmetic - which was measured on a real trace: two candidates agreeing
    to 13 significant figures (relative gap 4.1e-14) were separated by 2.3e-17
    and produced different reductions. Since RDP runs upstream of every
    detector, that noise propagates into the final knee."""

    @staticmethod
    def _symmetric_curve():
        # Exactly symmetric about its midpoint: the two shoulder points are
        # equidistant from the end-to-end chord by construction.
        x = np.arange(21, dtype=float)
        y = np.concatenate([np.linspace(1.0, 0.2, 10), [0.15],
                            np.linspace(0.2, 1.0, 10)])
        return np.column_stack((x, y))

    def test_split_is_invariant_to_last_bit_noise(self):
        base = self._symmetric_curve()
        rng = np.random.default_rng(0)
        shapes = set()
        for _ in range(100):
            pts = base.copy()
            pts[:, 1] += rng.uniform(-1e-15, 1e-15, len(pts))
            reduced, _ = rdp.rdp(pts, t=0.01)
            shapes.add(tuple(int(i) for i in reduced))
        self.assertEqual(len(shapes), 1)

    def test_a_genuinely_furthest_point_is_still_chosen(self):
        # The tolerance must not blur a real corner into a tie: this curve
        # bends hard at index 5 and nowhere else, so the reduction has to
        # keep that point.
        x = np.arange(11, dtype=float)
        y = np.concatenate([np.linspace(1.0, 0.2, 6), np.full(5, 0.2)])
        reduced, _ = rdp.rdp(np.column_stack((x, y)), t=0.001)
        self.assertIn(5, [int(i) for i in reduced])


class TestGlobalRDPIsMoreAggressive(unittest.TestCase):
    """`grdp` stops on the GLOBAL reconstruction cost rather than the current
    segment's, so it should reach a comparable fit with fewer points: `rdp`
    must split every locally-bad segment, while `grdp` can stop once the
    curve as a whole is good enough.

    Worth pinning, because it is easy to measure the opposite by accident.
    `smape` and `rpd` divide by the magnitude of the values, so on a curve
    decaying toward 0 the tail dominates the global cost however well it is
    fitted, and `grdp` keeps splitting it - inverting the comparison. The
    curve here is bounded away from zero, like the traces in `traces/`.
    """

    @staticmethod
    def _mrc_curve(n: int = 200) -> np.ndarray:
        x = np.arange(n, dtype=float)
        return np.column_stack((x, 0.1 + 0.9 * np.exp(-x / (n / 8.0))))

    def test_grdp_keeps_no_more_points_than_rdp(self):
        points = self._mrc_curve()
        for t in (0.001, 0.005, 0.01, 0.05):
            with self.subTest(t=t):
                local, _ = rdp.rdp(points, t=t)
                glob, _ = rdp.grdp(points, t=t)
                self.assertLessEqual(len(glob), len(local))

    def test_grdp_is_strictly_smaller_on_a_smooth_curve(self):
        # Needs a SMOOTH curve to show a strict gap. On a piecewise-linear one
        # both algorithms bottom out at the same optimum - start, corner, end -
        # and there is nothing left for a better stopping rule to win. It is a
        # continuously bending curve that rdp over-splits.
        points = self._mrc_curve()
        for t in (0.0005, 0.001, 0.01, 0.05):
            with self.subTest(t=t):
                local, _ = rdp.rdp(points, t=t)
                glob, _ = rdp.grdp(points, t=t)
                self.assertLess(len(glob), len(local))


class TestSplitOrdering(unittest.TestCase):
    """`grdp` explores the worst segment first, and these three heuristics are
    how it decides which that is. Each returns (left_cost, right_cost) for a
    split at `index`."""

    def setUp(self):
        # Bends at 7, dead flat after: splitting there must cost nothing on
        # the right and something on the left.
        y = np.concatenate([np.linspace(1.0, 0.3, 8), np.full(12, 0.3)])
        self.points = np.column_stack((np.arange(len(y), dtype=float), y))
        self.orderings = {
            'segment': lambda i: rdp.order_segment(self.points, i),
            'triangle': lambda i: rdp.order_triangle(self.points, i, lf.shortest_distance_points),
            'area': lambda i: rdp.order_area(self.points, i, lf.shortest_distance_points),
        }

    def test_a_flat_remainder_costs_nothing(self):
        for name, fn in self.orderings.items():
            with self.subTest(order=name):
                _, right = fn(8)
                self.assertAlmostEqual(right, 0.0)

    def test_the_bent_side_costs_more_than_the_flat_side(self):
        for name, fn in self.orderings.items():
            with self.subTest(order=name):
                left, right = fn(8)
                self.assertGreater(left, right)

    def test_splitting_at_the_corner_beats_splitting_in_the_tail(self):
        # A split at the corner leaves two well-fitted halves; one deep in the
        # tail leaves the whole bend on the left.
        for name, fn in self.orderings.items():
            with self.subTest(order=name):
                self.assertLess(sum(fn(7)), sum(fn(15)))


class TestComputeCostCoef(unittest.TestCase):
    def setUp(self):
        y = np.concatenate([np.linspace(1.0, 0.3, 8), np.full(12, 0.3)])
        self.points = np.column_stack((np.arange(len(y), dtype=float), y))

    def test_an_exact_fit_costs_nothing(self):
        x = np.arange(5, dtype=float)
        straight = np.column_stack((x, 2.0 * x + 1.0))
        for metric in (metrics.Metrics.smape, metrics.Metrics.rpd,
                       metrics.Metrics.rmspe, metrics.Metrics.rmsle):
            with self.subTest(metric=metric.value):
                self.assertAlmostEqual(rdp.compute_cost_coef(straight, (1.0, 2.0), metric), 0.0)

    def test_r2_scores_an_exact_fit_as_one(self):
        x = np.arange(5, dtype=float)
        straight = np.column_stack((x, 2.0 * x + 1.0))
        self.assertAlmostEqual(
            rdp.compute_cost_coef(straight, (1.0, 2.0), metrics.Metrics.r2), 1.0)

    def test_every_metric_is_dispatchable(self):
        coef = lf.linear_fit_points(self.points)
        for metric in metrics.Metrics:
            with self.subTest(metric=metric.value):
                self.assertTrue(np.isfinite(rdp.compute_cost_coef(self.points, coef, metric)))


class TestMinPointRDP(unittest.TestCase):
    """Tries successively tighter thresholds and returns as soon as one keeps
    at least `min_points`, falling back to the fixed-length version."""

    def setUp(self):
        y = np.concatenate([np.linspace(1.0, 0.3, 8), np.full(12, 0.3)])
        self.points = np.column_stack((np.arange(len(y), dtype=float), y))

    def test_it_keeps_at_least_the_requested_points(self):
        for min_points in (3, 5, 8):
            with self.subTest(min_points=min_points):
                reduced, _ = rdp.min_point_rdp(self.points, min_points=min_points)
                self.assertGreaterEqual(len(reduced), min_points)

    def test_it_keeps_the_endpoints(self):
        reduced, _ = rdp.min_point_rdp(self.points, min_points=5)
        self.assertEqual(int(reduced[0]), 0)
        self.assertEqual(int(reduced[-1]), len(self.points) - 1)

    def test_it_does_not_reorder_the_callers_thresholds(self):
        # Regression: this used to call t.sort(reverse=True), which reordered
        # the caller's own list in place - and with the mutable default it
        # once carried, that mutation persisted between calls.
        thresholds = [0.0001, 0.01, 0.001]
        before = list(thresholds)
        rdp.min_point_rdp(self.points, t=thresholds, min_points=5)
        self.assertEqual(thresholds, before)

    def test_thresholds_are_tried_largest_first_regardless_of_order(self):
        ascending, _ = rdp.min_point_rdp(self.points, t=[0.0001, 0.001, 0.01], min_points=5)
        descending, _ = rdp.min_point_rdp(self.points, t=[0.01, 0.001, 0.0001], min_points=5)
        np.testing.assert_array_equal(ascending, descending)


class TestPlotFrame(unittest.TestCase):
    """`plot_frame` renders one simplification step, for animating RDP.

    It had no tests and could not run: it called `plt` from an import that is
    commented out at module scope, so every call raised NameError. matplotlib
    is now imported inside the function, which keeps it an optional
    dependency - these tests skip when it is absent.
    """

    def setUp(self):
        try:
            import matplotlib
            matplotlib.use('Agg')          # headless: no display in CI
        except ImportError:
            self.skipTest('matplotlib is not installed')
        y = np.concatenate([np.linspace(1.0, 0.2, 10), np.full(20, 0.2)])
        self.points = np.column_stack((np.arange(len(y), dtype=float), y))
        self.reduced, _ = rdp.grdp(self.points, t=0.01)

    def test_writes_a_png_and_returns_its_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = rdp.plot_frame(self.points, self.reduced, 'frame', directory=d)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(path.endswith('img_frame.png'))
            self.assertGreater(os.path.getsize(path), 0)
            with open(path, 'rb') as f:
                self.assertEqual(f.read(4), b'\x89PNG')

    def test_creates_the_directory_if_it_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            nested = os.path.join(d, 'a', 'b')
            path = rdp.plot_frame(self.points, self.reduced, 0, directory=nested)
            self.assertTrue(os.path.exists(path))

    def test_each_name_gets_its_own_file(self):
        with tempfile.TemporaryDirectory() as d:
            paths = {rdp.plot_frame(self.points, self.reduced, i, directory=d)
                     for i in range(3)}
            self.assertEqual(len(paths), 3)
            self.assertEqual(len(os.listdir(d)), 3)


if __name__ == '__main__':
    unittest.main()