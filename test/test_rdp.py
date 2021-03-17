import unittest
import numpy as np

from knee.rdp import straight_line


class TestRDP(unittest.TestCase):
    def test_straight_line_95(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 4.0],
                           [5.0, 3.0], [6.0, 2.0], [7.0, 1.0], [8.0, 0.0], [9.0, 0.0]])
        result = straight_line(points, 0, 7, .95)
        desired = 1
        self.assertEqual(result, desired)

    def test_straight_line_90(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 4.0],
                           [5.0, 3.0], [6.0, 2.0], [7.0, 1.0], [8.0, 0.0], [9.0, 0.0]])
        result = straight_line(points, 0, 7, .90)
        desired = 0
        self.assertEqual(result, desired)

    def test_straight_line_99(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 4.0],
                           [5.0, 3.0], [6.0, 2.0], [7.0, 1.0], [8.0, 0.0], [9.0, 0.0]])
        result = straight_line(points, 0, 7, .99)
        desired = 3
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