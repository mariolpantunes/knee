import unittest
import numpy as np
import knee.rdp as rdp


class TestRDP(unittest.TestCase):
    
    def test_rdp_0(self):
        points = np.array([[1, 5], [2, 5], [3, 5], [4, 5], [5, 5]])
        reduced, removed = rdp.rdp(points)
        desired = np.array([0, 4])
        np.testing.assert_array_equal(reduced, desired)
        desired = np.array([[0,3]])
        np.testing.assert_array_equal(removed, desired)
    
    def test_rdp_1(self):
        points = np.array([[1, 5], [2, 5], [3, 6], [4, 6], [5, 6]])
        reduced, removed = rdp.rdp(points)
        desired = np.array([0, 1, 2, 4])
        np.testing.assert_array_equal(reduced, desired)
        desired = np.array([[0,0],[1, 0],[2,1]])
        np.testing.assert_array_equal(removed, desired)

    def test_rdp_mapping_line(self):
        points = np.array([[1, 5], [2, 5], [3, 5], [4, 5], [5, 5]])
        reduced, removed = rdp.rdp(points)
        indexes = np.array([0, 1])
        result = rdp.mapping(indexes, reduced, removed)
        desired = np.array([0, 4])
        np.testing.assert_array_equal(result, desired)
    
    def test_rdp_mapping_two(self):
        points = np.array([[0, 3], [1, 3], [2, 3], [3, 2], [4, 1], [5, 0]])
        reduced, removed = rdp.rdp(points)
        indexes = np.array([0, 1, 2])
        result = rdp.mapping(indexes, reduced, removed)
        desired = np.array([0, 2, 5])
        np.testing.assert_array_equal(result, desired)
    
    def test_rdp_mapping_four(self):
        points = np.array([[2, 0], [3, 1], [4, 2], [5, 2], [6, 2], [7, 3], [8, 4], [9, 3], [10, 2], [11, 1], [12, 0]])
        reduced, removed = rdp.rdp(points)
        indexes = np.array([0, 1, 2, 3, 4])
        result = rdp.mapping(indexes, reduced, removed)
        desired = np.array([0, 2, 4, 6, 10])
        np.testing.assert_array_equal(result, desired)
