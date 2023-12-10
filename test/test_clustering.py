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
import knee.clustering as clustering


class TestClustering(unittest.TestCase):
    def test_single_linkage_two(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [7.0, 1.0], [8.0, 0.0], [9.0, 0.0]])
        result = clustering.single_linkage(points, 0.2)
        desired = np.array([0,0,0,1,1,1])
        np.testing.assert_array_equal(result, desired)
    
    def test_single_linkage_three(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [5.0, 5.0], [8.0, 0.0], [9.0, 0.0]])
        result = clustering.single_linkage(points, 0.2)
        desired = np.array([0,0,1,2,2])
        np.testing.assert_array_equal(result, desired)
    
    def test_single_linkage_individual(self):
        points = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0], [7.0, 0.0], [9.0, 0.0]])
        result = clustering.single_linkage(points, 0.2)
        desired = np.array([0,1,2,3,4])
        np.testing.assert_array_equal(result, desired)
    
    def test_complete_linkage_two(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [7.0, 1.0], [8.0, 0.0], [9.0, 0.0]])
        result = clustering.complete_linkage(points, 0.2)
        desired = np.array([0,0,1,2,2,3])
        np.testing.assert_array_equal(result, desired)
    
    def test_complete_linkage_three(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [5.0, 5.0], [8.0, 0.0], [9.0, 0.0]])
        result = clustering.complete_linkage(points, 0.2)
        desired = np.array([0,0,1,2,2])
        np.testing.assert_array_equal(result, desired)
    
    def test_complete_linkage_individual(self):
        points = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0], [7.0, 0.0], [9.0, 0.0]])
        result = clustering.complete_linkage(points, 0.2)
        desired = np.array([0,1,2,3,4])
        np.testing.assert_array_equal(result, desired)
    
    def test_centroid_linkage_two(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [7.0, 1.0], [8.0, 0.0], [9.0, 0.0]])
        result = clustering.centroid_linkage(points, 0.2)
        desired = np.array([0,0,0,1,1,1])
        np.testing.assert_array_equal(result, desired)
    
    def test_centroid_linkage_three(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [5.0, 5.0], [8.0, 0.0], [9.0, 0.0]])
        result = clustering.centroid_linkage(points, 0.2)
        desired = np.array([0,0,1,2,2])
        np.testing.assert_array_equal(result, desired)
    
    def test_centroid_linkage_individual(self):
        points = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0], [7.0, 0.0], [9.0, 0.0]])
        result = clustering.centroid_linkage(points, 0.2)
        desired = np.array([0,1,2,3,4])
        np.testing.assert_array_equal(result, desired)

    def test_average_linkage_two(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [7.0, 1.0], [8.0, 0.0], [9.0, 0.0]])
        result = clustering.average_linkage(points, 0.2)
        desired = np.array([0,0,0,1,1,1])
        np.testing.assert_array_equal(result, desired)
    
    def test_average_linkage_three(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [5.0, 5.0], [8.0, 0.0], [9.0, 0.0]])
        result = clustering.average_linkage(points, 0.2)
        desired = np.array([0,0,1,2,2])
        np.testing.assert_array_equal(result, desired)
    
    def test_average_linkage_individual(self):
        points = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0], [7.0, 0.0], [9.0, 0.0]])
        result = clustering.average_linkage(points, 0.2)
        desired = np.array([0,1,2,3,4])
        np.testing.assert_array_equal(result, desired)


if __name__ == '__main__':
    unittest.main()