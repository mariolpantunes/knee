# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''

import unittest
import numpy as np
import knee.metrics as metrics


class TestMetrics(unittest.TestCase):
    def test_rpd_00(self):
        y = np.array([0, 0, 1, 2, 2, 2, 2, 3, 4])
        y_hat = np.array([0, 0, 1, 2, 2, 2, 2, 3, 4])
        result = metrics.rpd(y, y_hat)
        desired = 0.0
        self.assertAlmostEqual(result, desired, 2)
    
    def test_rpd_01(self):
        y = np.array([0, 0, 0, 1, 2, 3, 3, 3, 2, 2, 2, 1, 0, 0])
        y_hat = np.array([0, 0, 0, 1, 2, 3, 3, 3, 2, 2, 2, 1, 0, 0])
        result = metrics.rpd(y, y_hat)
        desired = 0.0
        self.assertAlmostEqual(result, desired, 2)

    def test_rmsle_00(self):
        y = np.array([0, 0, 1, 2, 2, 2, 2, 3, 4])
        y_hat = np.array([0, 0, 1, 2, 2, 2, 2, 3, 4])
        result = metrics.rmsle(y, y_hat)
        desired = 0.0
        self.assertAlmostEqual(result, desired, 2)
    
    def test_rmsle_01(self):
        y = np.array([0, 0, 0, 1, 2, 3, 3, 3, 2, 2, 2, 1, 0, 0])
        y_hat = np.array([0, 0, 0, 1, 2, 3, 3, 3, 2, 2, 2, 1, 0, 0])
        result = metrics.rmsle(y, y_hat)
        desired = 0.0
        self.assertAlmostEqual(result, desired, 2)

    def test_rmspe_00(self):
        y = np.array([0, 0, 1, 2, 2, 2, 2, 3, 4])
        y_hat = np.array([0, 0, 1, 2, 2, 2, 2, 3, 4])
        result = metrics.rmspe(y, y_hat)
        desired = 0.0
        self.assertAlmostEqual(result, desired, 2)
    
    def test_rmspe_01(self):
        y = np.array([0, 0, 0, 1, 2, 3, 3, 3, 2, 2, 2, 1, 0, 0])
        y_hat = np.array([0, 0, 0, 1, 2, 3, 3, 3, 2, 2, 2, 1, 0, 0])
        result = metrics.rmspe(y, y_hat)
        desired = 0.0
        self.assertAlmostEqual(result, desired, 2)

    def test_r2_00(self):
        y = np.array([0, 0, 1, 2, 2, 2, 2, 3, 4])
        y_hat = np.array([0, 0, 1, 2, 2, 2, 2, 3, 4])
        result = metrics.r2(y, y_hat)
        desired = 1.0
        self.assertAlmostEqual(result, desired, 2)

    def test_r2_01(self):
        y = np.array([0, 0, 0, 1, 2, 3, 3, 3, 2, 2, 2, 1, 0, 0])
        y_hat = np.array([0, 0, 0, 1, 2, 3, 3, 3, 2, 2, 2, 1, 0, 0])
        result = metrics.r2(y, y_hat)
        desired = 1.0
        self.assertAlmostEqual(result, desired, 2)

    def test_smape_00(self):
        y = np.array([0, 0, 1, 2, 2, 2, 2, 3, 4])
        y_hat = np.array([0, 0, 1, 2, 2, 2, 2, 3, 4])
        result = metrics.smape(y, y_hat)
        desired = 0.0
        self.assertAlmostEqual(result, desired, 2)
    
    def test_smape_01(self):
        y = np.array([0, 0, 0, 1, 2, 3, 3, 3, 2, 2, 2, 1, 0, 0])
        y_hat = np.array([0, 0, 0, 1, 2, 3, 3, 3, 2, 2, 2, 1, 0, 0])
        result = metrics.smape(y, y_hat)
        desired = 0.0
        self.assertAlmostEqual(result, desired, 2)


if __name__ == '__main__':
    unittest.main()