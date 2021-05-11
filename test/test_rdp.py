import unittest
import numpy as np

from knee.rdp import rdp, mapping, straight_line


class TestRDP(unittest.TestCase):

    def test_rdp_mapping_line(self):
        points = np.array([[1, 5], [2, 5], [3, 5], [4, 5], [5, 5]])
        points_reduced, removed = rdp(points)
        
        indexes = np.array([0, 1])
        result = mapping(indexes, points_reduced, removed)
        desired = np.array([0, 4])
        np.testing.assert_array_equal(result, desired)
    
    def test_rdp_mapping_two(self):
        points = np.array([[0, 3], [1, 3], [2, 3], [3, 2], [4, 1], [5, 0]])
        points_reduced, removed = rdp(points)
        
        indexes = np.array([0, 1, 2])
        result = mapping(indexes, points_reduced, removed)
        desired = np.array([0, 2, 5])
        np.testing.assert_array_equal(result, desired)
    
    def test_rdp_mapping_four(self):
        points = np.array([[2, 0], [3, 1], [4, 2], [5, 2], [6, 2], [7, 3], [8, 4], [9, 3], [10, 2], [11, 1], [12, 0]])
        points_reduced, removed = rdp(points)
        
        indexes = np.array([0, 1, 2, 3, 4])
        result = mapping(indexes, points_reduced, removed)
        desired = np.array([0, 2, 4, 6, 10])
        np.testing.assert_array_equal(result, desired)

    def test_straight_line_95(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 4.0],
                           [5.0, 3.0], [6.0, 2.0], [7.0, 1.0], [8.0, 0.0], [9.0, 0.0]])
        result = straight_line(points, 0, 7, .95)
        desired = 2
        self.assertEqual(result, desired)

    def test_straight_line_90(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 4.0],
                           [5.0, 3.0], [6.0, 2.0], [7.0, 1.0], [8.0, 0.0], [9.0, 0.0]])
        result = straight_line(points, 0, 7, .90)
        desired = 1
        self.assertEqual(result, desired)

    def test_straight_line_99(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 4.0],
                           [5.0, 3.0], [6.0, 2.0], [7.0, 1.0], [8.0, 0.0], [9.0, 0.0]])
        result = straight_line(points, 0, 7, .99)
        desired = 2
        self.assertEqual(result, desired)

    def test_one_point(self):
        points = np.array([[1.0, 5.0]])
        result = straight_line(points, 0, 0, .8)
        desired = 0
        self.assertEqual(result, desired)

    def test_two_point(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0]])
        result = straight_line(points, 0, 1, .8)
        desired = 0
        self.assertEqual(result, desired)
    
    def test_four_point(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 4.0]])
        result = straight_line(points, 0, 3, .9)
        desired = 2
        self.assertEqual(result, desired)
