import unittest
import numpy as np
import knee.metrics as metrics


class TestMetrics(unittest.TestCase):
    def test_rpd_00(self):
        y = np.array([0, 0, 1, 2, 2, 2, 2, 3, 4])
        y_hat = np.array([0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 4.0])
        result = metrics.rpd(y, y_hat)
        desired = 0.0
        self.assertAlmostEqual(result, desired, 2)