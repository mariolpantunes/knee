
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

from kneeliverse import menger


class TestMengerCurvature(unittest.TestCase):
    """Menger curvature is 4A/(|fg||gh||hf|) - the reciprocal of the radius
    of the circle through the three points."""

    def test_collinear_points_have_no_curvature(self):
        # The defining property, and the one that was broken: the numerator's
        # fabs covered only its first term, so these scored 1.0607.
        for triple in ([[1., 1.], [0., 0.], [2., 2.]],
                       [[5., 0.], [0., 0.], [9., 0.]],
                       [[0., 3.], [0., 1.], [0., 7.]]):
            with self.subTest(triple=triple):
                f, g, h = (np.array(p) for p in triple)
                self.assertAlmostEqual(menger.menger_curvature(f, g, h), 0.0)

    def test_it_is_the_reciprocal_of_the_circumradius(self):
        # Three points on the unit circle: curvature must be 1/R = 1.
        angles = [0.0, 2 * math.pi / 3, 4 * math.pi / 3]
        p = [np.array([math.cos(a), math.sin(a)]) for a in angles]
        self.assertAlmostEqual(menger.menger_curvature(p[0], p[1], p[2]), 1.0)

    def test_a_larger_circle_curves_less(self):
        for radius in (0.5, 1.0, 2.0, 10.0):
            with self.subTest(radius=radius):
                angles = [0.0, 2 * math.pi / 3, 4 * math.pi / 3]
                p = [np.array([radius * math.cos(a), radius * math.sin(a)]) for a in angles]
                self.assertAlmostEqual(menger.menger_curvature(p[0], p[1], p[2]), 1.0 / radius)

    def test_it_is_never_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            f, g, h = (rng.uniform(-5, 5, 2) for _ in range(3))
            self.assertGreaterEqual(menger.menger_curvature(f, g, h), 0.0)

    def test_point_order_does_not_change_the_magnitude(self):
        # The circle through three points does not depend on how they are
        # listed, so neither does its radius.
        f, g, h = np.array([0., 0.]), np.array([1., 2.]), np.array([3., 1.])
        expected = menger.menger_curvature(f, g, h)
        for triple in ((g, h, f), (h, f, g), (h, g, f)):
            with self.subTest(order=triple):
                self.assertAlmostEqual(menger.menger_curvature(*triple), expected)


class TestMengerKnee(unittest.TestCase):
    def _corner_curve(self, corner: int, n: int = 30) -> np.ndarray:
        y = np.concatenate([np.linspace(1.0, 0.3, corner + 1), np.full(n - corner - 1, 0.3)])
        return np.column_stack((np.arange(len(y), dtype=float), y))

    def test_it_finds_the_corner(self):
        # Before the numerator was fixed this returned 1 for every curve,
        # wherever the corner actually was.
        for corner in (5, 10, 15):
            with self.subTest(corner=corner):
                self.assertEqual(int(menger.knee(self._corner_curve(corner))), corner)

    def test_it_agrees_with_the_other_detectors(self):
        from kneeliverse import curvature, dfdt, lmethod
        points = self._corner_curve(9)
        expected = int(menger.knee(points))
        for name, fn in (('curvature', curvature.knee), ('dfdt', dfdt.knee),
                         ('lmethod', lmethod.knee)):
            with self.subTest(detector=name):
                self.assertLessEqual(abs(int(fn(points)) - expected), 1)

    def test_a_straight_line_has_no_corner_to_find(self):
        # Every triple is collinear, so every score is 0 and the tie resolves
        # to the leftmost - deterministically, rather than on noise.
        x = np.arange(20, dtype=float)
        points = np.column_stack((x, 2.0 * x + 1.0))
        self.assertEqual(int(menger.knee(points)), 0)

    def test_the_index_is_inside_the_curve(self):
        points = self._corner_curve(9)
        idx = int(menger.knee(points))
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, len(points))

    def test_multi_knee_returns_valid_indexes(self):
        points = self._corner_curve(9)
        knees = np.asarray(menger.multi_knee(points))
        self.assertTrue(np.all(knees >= 0))
        self.assertTrue(np.all(knees < len(points)))


class TestMengerDeterminism(unittest.TestCase):
    """Equally bent triples give values that agree mathematically and differ
    in their last bits; an exact argmax lets that noise choose the knee."""

    @staticmethod
    def _two_equal_bends():
        # Symmetric about the middle, so both bends score identically by
        # construction and the choice between them is a real tie.
        y = np.concatenate([np.linspace(1.0, 0.5, 6),
                            np.linspace(0.5, 1.0, 6)[1:]])
        return np.column_stack((np.arange(len(y), dtype=float), y))

    def test_knee_is_invariant_to_last_bit_noise(self):
        base = self._two_equal_bends()
        rng = np.random.default_rng(0)
        picks = set()
        for _ in range(100):
            points = base.copy()
            points[:, 1] += rng.uniform(-1e-15, 1e-15, len(points))
            picks.add(int(menger.knee(points)))
        self.assertEqual(len(picks), 1)

    def test_a_straight_run_scores_zero_throughout(self):
        # The run between the endpoints is collinear, so every interior point
        # on it must score exactly 0 - which is what makes the corner stand
        # out. It used to score a uniform 0.136 instead.
        y = np.concatenate([np.linspace(1.0, 0.3, 6), np.full(6, 0.3)])
        points = np.column_stack((np.arange(len(y), dtype=float), y))
        scores = [menger.menger_curvature(points[i], points[i - 1], points[i + 1])
                  for i in range(1, len(points) - 1)]
        self.assertAlmostEqual(scores[0], 0.0)   # inside the slope
        self.assertAlmostEqual(scores[1], 0.0)
        self.assertGreater(scores[4], 0.0)       # the corner at index 5


if __name__ == '__main__':
    unittest.main()
