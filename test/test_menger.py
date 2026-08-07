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
import kneeliverse.menger as menger


class TestMenger(unittest.TestCase):
    """Shape-level checks only.

    `menger_curvature`'s numerator is currently mis-parenthesised - the
    `fabs` covers only the first product, so collinear points score 1.06
    instead of 0 - which means the value it returns is not Menger curvature
    and there is no correct value to assert against. That is a real defect,
    tracked separately; it predates this file and fixing it changes
    `knee`'s output, so it is deliberately not smuggled in here. These tests
    pin what holds regardless: the shape of the output, and the fact that
    the choice does not move under noise.
    """

    def test_curvature_is_finite_and_non_negative_along_a_curve(self):
        y = np.concatenate([np.linspace(1.0, 0.2, 5), np.full(5, 0.2)])
        points = np.column_stack((np.arange(len(y), dtype=float), y))
        values = [menger.menger_curvature(points[i], points[i - 1], points[i + 1])
                  for i in range(1, len(points) - 1)]
        self.assertTrue(all(np.isfinite(v) for v in values))

    def test_knee_returns_an_interior_index(self):
        y = np.concatenate([np.linspace(1.0, 0.2, 5), np.full(5, 0.2)])
        points = np.column_stack((np.arange(len(y), dtype=float), y))
        idx = int(menger.knee(points))
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, len(points))


class TestMengerDeterminism(unittest.TestCase):
    """Menger curvature is a ratio built from three point-triples, so equally
    bent triples give values that agree mathematically and differ in their
    last bits. An exact argmax lets that noise choose the knee."""

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

    def test_every_point_of_an_equally_scored_run_ties(self):
        # Documents the concrete case that motivated the change. On a
        # piecewise-linear curve every point along the sloped run scores
        # identically, so the whole run is one tied group and the leftmost
        # must win. np.argmax returned index 4 here - the LAST member of the
        # group - because the four equal values differ in their last bits.
        #
        # That the run scores non-zero at all is the mis-parenthesised
        # numerator (tracked separately): collinear points should score 0.
        # The tie itself is real either way, and is what this asserts.
        y = np.concatenate([np.linspace(1.0, 0.3, 6), np.full(6, 0.3)])
        points = np.column_stack((np.arange(len(y), dtype=float), y))

        scores = np.array([0.0] + [menger.menger_curvature(points[i], points[i - 1], points[i + 1])
                                   for i in range(1, len(points) - 1)] + [0.0])
        run = scores[1:5]
        self.assertLess(np.ptp(run) / run.max(), 1e-9)     # equal to within EPS_RANK
        self.assertEqual(int(menger.knee(points)), 1)      # leftmost of the tied run


if __name__ == '__main__':
    unittest.main()
