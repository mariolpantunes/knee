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

if __name__ == '__main__':
    unittest.main()