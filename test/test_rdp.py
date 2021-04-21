import unittest
import numpy as np

from knee.rdp import naive_straight_line, straight_line


class TestRDP(unittest.TestCase):
    def test_straight_line_95(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 4.0],
                           [5.0, 3.0], [6.0, 2.0], [7.0, 1.0], [8.0, 0.0], [9.0, 0.0]])
        result = straight_line(points, 0, 7, .95)
        #naive_straight_line(points, 0, 7, .90)
        desired = 2
        self.assertEqual(result, desired)

    def test_straight_line_90(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 4.0],
                           [5.0, 3.0], [6.0, 2.0], [7.0, 1.0], [8.0, 0.0], [9.0, 0.0]])
        result = straight_line(points, 0, 7, .90)
        #naive_straight_line(points, 0, 7, .90)
        desired = 1
        self.assertEqual(result, desired)

    def test_straight_line_99(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 4.0],
                           [5.0, 3.0], [6.0, 2.0], [7.0, 1.0], [8.0, 0.0], [9.0, 0.0]])
        result = straight_line(points, 0, 7, .99)
        #naive_straight_line(points, 0, 7, .99)
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
    
    def test_binary_naive_80(self):
        points = np.array([[3., 0.84], [11., 0.8], [26., 0.77], [47., 0.76], [92., 0.75], [174., 0.73], [249., 0.73], [386., 0.72]])
        result = straight_line(points, 0, 7, .8)
        desired = naive_straight_line(points, 0, 7, .8)
        #print(f'[{result}, {desired}]')
        self.assertEqual(result, desired)
    
    def test_binary_naive_90(self):
        points = np.array([[3., 0.84], [11., 0.8], [26., 0.77], [47., 0.76], [92., 0.75], [174., 0.73], [249., 0.73], [386., 0.72]])
        result = straight_line(points, 0, 7, .9)
        desired = naive_straight_line(points, 0, 7, .9)
        #print(f'[{result}, {desired}]')
        self.assertEqual(result, desired)
    
    def test_binary_naive_2(self):
        points = np.array([[65216, 0.629], [65217, 0.57], [65371, 0.57], [65597, 0.57], [65885, 0.57], [65886, 0.561]])
        result = straight_line(points, 0, 3, .8)
        desired = naive_straight_line(points, 0, 3, .8)
        print(f'[{result}, {desired}]')
        self.assertEqual(result, desired)
