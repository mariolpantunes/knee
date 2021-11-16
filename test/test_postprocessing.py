import unittest
import numpy as np
import knee.rdp as rdp
import knee.postprocessing as pp


class TestPostProcessing(unittest.TestCase):
    
    def test_filter_corner_knees(self):
        points = np.array([[1,3], [2,3], [3,3], [4,2.5], [5,2], [6,1.5], [7, 1]])
        knees = np.array([1,2,3])
        result = pp.filter_corner_knees(points, knees, .5)
        desired = np.array([1,3])
        np.testing.assert_array_equal(result, desired)
    
    def test_add_even_points_0(self):
        points = np.array([[0, 6], [1, 5], [2, 4], [3, 1], [4, 2], [5, 2], [6, 2], [7, 3], [8, 3], [9, 2], [10, 1], [11, 1/4], [12, 0]])
        knees = np.array([1,2])

        points_reduced, removed = rdp.rdp(points)

        result = pp.add_points_even(points, points_reduced, knees, removed,  extremes=False)
        desired = np.array([1,2,3,10,11,12])
        np.testing.assert_array_equal(result, desired)
    
    def test_add_even_points_1(self):
        points = np.array([[0, 6], [1, 5], [2, 4], [3, 1], [4, 2], [5, 2], [6, 2], [7, 3], [8, 3], [9, 2], [10, 1], [11, 1/4], [12, 0]])
        knees = np.array([1,2])

        points_reduced, removed = rdp.rdp(points)

        result = pp.add_points_even(points, points_reduced, knees, removed,  extremes=True)
        desired = np.array([0,1,2,3,10,11,12])
        np.testing.assert_array_equal(result, desired)