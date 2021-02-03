import unittest
import numpy as np

from knee.rdp import strait_line

class TestRDP(unittest.TestCase):
    def test_strait_line_95(self):
        points = np.array([[1.0,5.0], [2.0,5.0], [3.0, 5.0],[4.0,4.0],
        [5.0,3.0], [6.0, 2.0], [7.0,1.0], [8.0, 0.0], [9.0, 0.0]])
        result = strait_line(points, 0, 7, .95)
        desired = 1
        self.assertEqual(result, desired)
    
    def test_strait_line_90(self):
        points = np.array([[1.0,5.0], [2.0,5.0], [3.0, 5.0],[4.0,4.0],
        [5.0,3.0], [6.0, 2.0], [7.0,1.0], [8.0, 0.0], [9.0, 0.0]])
        result = strait_line(points, 0, 7, .90)
        desired = 0
        self.assertEqual(result, desired)
    
    def test_strait_line_90(self):
        points = np.array([[1.0,5.0], [2.0,5.0], [3.0, 5.0],[4.0,4.0],
        [5.0,3.0], [6.0, 2.0], [7.0,1.0], [8.0, 0.0], [9.0, 0.0]])
        result = strait_line(points, 0, 7, .99)
        desired = 2
        self.assertEqual(result, desired)