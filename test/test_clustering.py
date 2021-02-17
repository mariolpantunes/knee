import unittest
import numpy as np


from knee.clustering import single_linkage


class TestClustering(unittest.TestCase):
    def test_single_linkage_two(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [7.0, 1.0], [8.0, 0.0], [9.0, 0.0]])
        d = points[-1,0] - points[0,0]
        result = single_linkage(points, d)
        print(result)
        desired = np.array([0,0,0,1,1,1])
        np.testing.assert_array_equal(result, desired)
    
    def test_single_linkage_three(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [5.0, 5.0], [8.0, 0.0], [9.0, 0.0]])
        d = points[-1,0] - points[0,0]
        result = single_linkage(points, d)
        print(result)
        desired = np.array([0,0,1,2,2])
        np.testing.assert_array_equal(result, desired)
    
    def test_single_linkage_individual(self):
        points = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0], [7.0, 0.0], [9.0, 0.0]])
        d = points[-1,0] - points[0,0]
        result = single_linkage(points, d)
        print(result)
        desired = np.array([0,1,2,3,4])
        np.testing.assert_array_equal(result, desired)