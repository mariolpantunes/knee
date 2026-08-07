
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

from kneeliverse import curvature


class TestCurvature(unittest.TestCase):
    def test_get_knee(self):
        x = np.array([0,1,2,3,4,5,6,7,8,9])
        y = np.array([1,0.5,0.333333333,0.25,0.2,0.166666667,0.142857143,0.125,0.111111111,0.1])
        points = np.stack((x, y), axis=1)
        result = curvature.knee(points)
        desired = 1
        self.assertEqual(result, desired)
    
    def test_multi_knee(self):
        x = np.array([0,1,2,3,4,5,6,7,8,9])
        y = np.array([1,0.5,0.333333333,0.25,0.2,0.2,0.1,0.06666666667,0.05,0.04])
        points = np.stack((x, y), axis=1)
        result = curvature.multi_knee(points)
        desired = np.array([1, 4, 5, 7])
        np.testing.assert_array_equal(result, desired)


class TestCurvatureDeterminism(unittest.TestCase):
    """Curvature is built from two finite-difference gradients, so a curve
    that bends equally in two places produces values that match
    mathematically and differ in their last bits. An exact argmax then lets
    that noise pick the knee."""

    @staticmethod
    def _two_equal_bends():
        # Symmetric about the middle: the two bends have identical curvature
        # by construction, so the choice between them is a genuine tie.
        y = np.concatenate([np.linspace(1.0, 0.5, 6),
                            np.linspace(0.5, 1.0, 6)[1:]])
        return np.column_stack((np.arange(len(y), dtype=float), y))

    def test_knee_is_invariant_to_last_bit_noise(self):
        base = self._two_equal_bends()
        rng = np.random.default_rng(0)
        picks = set()
        for _ in range(100):
            pts = base.copy()
            pts[:, 1] += rng.uniform(-1e-15, 1e-15, len(pts))
            picks.add(int(curvature.knee(pts)))
        self.assertEqual(len(picks), 1)

    def test_a_genuinely_sharper_bend_still_wins(self):
        # One hard corner at index 4 and a gentle one later: the tolerance
        # must not merge them into a tie and default to the leftmost.
        y = np.concatenate([np.linspace(1.0, 0.3, 5), np.full(3, 0.3),
                            np.linspace(0.3, 0.28, 4)])
        pts = np.column_stack((np.arange(len(y), dtype=float), y))
        self.assertEqual(int(curvature.knee(pts)), 4)


if __name__ == '__main__':
    unittest.main()