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