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

    def test_aip(self):
        points = np.array([[1,.5],[2,.33],[3,.25],[4,.2], [5,.16], [6, 0.14], [7, .125], [8,.11], [9,.1]])
        reduced = np.array([0, 1, 4, 8])
        result = evaluation.aip(points, reduced)
        desired = 0.03145264095570805
        self.assertAlmostEqual(result, desired)



if __name__ == '__main__':
    unittest.main()