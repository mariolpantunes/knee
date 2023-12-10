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

import math
import unittest
import numpy as np
import knee.evaluation as evaluation


class TestEvaluation(unittest.TestCase):
    def test_mae_0(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0,1,2])
        expected = np.array([[0,0], [1,1], [2,2]])
        result = evaluation.mae(points, knees, expected)
        desired = 0.0
        self.assertAlmostEqual(result, desired)
    
    def test_mae_1(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0,1,2])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.mae(points, knees, expected, evaluation.Strategy.worst)
        desired = 1/3
        self.assertAlmostEqual(result, desired)
    
    def test_mae_2(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([1])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.mae(points, knees, expected)
        desired = 1/2
        self.assertAlmostEqual(result, desired)
    
    def test_mse_0(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0,1,2])
        expected = np.array([[0,0], [1,1], [2,2]])
        result = evaluation.mse(points, knees, expected)
        desired = 0.0
        self.assertAlmostEqual(result, desired)
    
    def test_mse_1(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0,1,2])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.mse(points, knees, expected, evaluation.Strategy.worst)
        desired = 1/3
        self.assertAlmostEqual(result, desired)
    
    def test_mse_2(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([1])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.mse(points, knees, expected)
        desired = 1/2
        self.assertAlmostEqual(result, desired)
    
    def test_rmse_0(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0,1,2])
        expected = np.array([[0,0], [1,1], [2,2]])
        result = evaluation.rmse(points, knees, expected)
        desired = 0.0
        self.assertAlmostEqual(result, desired)
    
    def test_rmse_1(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0,1,2])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.rmse(points, knees, expected, evaluation.Strategy.worst)
        desired = math.sqrt(1/3)
        self.assertAlmostEqual(result, desired)
    
    def test_rmse_2(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([1])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.rmse(points, knees, expected)
        desired = math.sqrt(1/2)
        self.assertAlmostEqual(result, desired)
    
    def test_rmspe_0(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0,1,2])
        expected = np.array([[0,0], [1,1], [2,2]])
        result = evaluation.rmspe(points, knees, expected)
        desired = 0.0
        self.assertAlmostEqual(result, desired)
    
    def test_rmspe_1(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0,1,2])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.rmspe(points, knees, expected, evaluation.Strategy.worst)
        desired = 5773502691896258.0
        self.assertAlmostEqual(result, desired, 3)
    
    def test_rmspe_2(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([1])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.rmspe(points, knees, expected)
        desired = 0.3535533905755961
        self.assertAlmostEqual(result, desired, 3)
    
    def test_cm_0(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([1])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.cm(points, knees, expected)
        desired = np.array([[1,0],[1,1]])
        np.testing.assert_array_equal(result, desired)
    
    def test_cm_1(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([1])
        expected = np.array([[0,0], [2,2]])
        result = evaluation.cm(points, knees, expected, t=.5)
        desired = np.array([[1,0],[1,1]])
        np.testing.assert_array_equal(result, desired)
    
    def test_cm_2(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.cm(points, knees, expected)
        desired = np.array([[0,1],[2,0]])
        np.testing.assert_array_equal(result, desired)
    
    def test_mcc_0(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([1])
        expected = np.array([[1,1], [2,2]])
        cm = evaluation.cm(points, knees, expected)
        result = evaluation.mcc(cm)
        desired = 0.5
        self.assertAlmostEqual(result, desired, 3)
    
    def test_mcc_1(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([1])
        expected = np.array([[0,0], [2,2]])
        cm = evaluation.cm(points, knees, expected, t=.5)
        result = evaluation.mcc(cm)
        desired = 0.5
        self.assertAlmostEqual(result, desired)
    
    def test_mcc_2(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0])
        expected = np.array([[1,1], [2,2]])
        cm = evaluation.cm(points, knees, expected)
        result = evaluation.mcc(cm)
        desired = -1.0
        self.assertAlmostEqual(result, desired)

    def test_compute_global_rmse_0(self):
        points = np.array([[0,2], [1,1], [2,0], [3,1], [4,2]])
        reduced = np.array([0,2,4])
        result = evaluation.compute_global_rmse(points, reduced)
        desired = 0.0
        self.assertAlmostEqual(result, desired)
    
    def test_compute_global_rmse_1(self):
        points = np.array([[0,2], [1,1], [2,0], [3,1], [4,2]])
        reduced = np.array([0,1,3,4])
        result = evaluation.compute_global_rmse(points, reduced)
        desired = 0.4472135954999579
        self.assertAlmostEqual(result, desired)
    
    def test_mip_0(self):
        points = np.array([[0,2], [1,1], [2,0], [3,1], [4,2]])
        reduced = np.array([0,2,4])
        mip, _ = evaluation.mip(points, reduced)
        desired = 1.0954451150103321
        self.assertAlmostEqual(mip, desired)

    def test_mip_1(self):
        points = np.array([[0,2], [1,1], [2,0], [3,1], [4,2]])
        reduced = np.array([0,1,3,4])
        mip, _ = evaluation.mip(points, reduced)
        desired = 0.2194530711667088
        self.assertAlmostEqual(mip, desired)

    def test_compute_global_cost_0(self):
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 2], [4, 3], [5, 4]])
        reduced = np.array([0, 2, 3, 5])
        result = evaluation.compute_global_cost(points, reduced)
        desired = 0.0
        self.assertEqual(result, desired)
    
    def test_compute_global_cost_1(self):
        points = np.array([[0, 2], [0, 1], [1/2, 1/2], [1, 0], [2, 0]])
        reduced = np.array([0, 1, 3, 4])
        result = evaluation.compute_global_cost(points, reduced)
        #desired = 0.2857142857142857
        desired = 0.0
        self.assertEqual(result, desired)


if __name__ == '__main__':
    unittest.main()