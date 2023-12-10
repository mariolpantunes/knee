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
import knee.evaluation as evaluation


class TestRDP(unittest.TestCase):
    def test_get_neighbourhood_90(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 4.0],
                           [5.0, 3.0], [6.0, 2.0], [7.0, 1.0], [8.0, 0.0], [9.0, 0.0]])
        idx, _, _ = evaluation.get_neighbourhood_points(points, 7, 0, .90)
        desired = 1
        self.assertEqual(idx, desired)
    
    def test_get_neighbourhood_99(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 4.0],
                           [5.0, 3.0], [6.0, 2.0], [7.0, 1.0], [8.0, 0.0], [9.0, 0.0]])
        idx, _, _ = evaluation.get_neighbourhood_points(points, 7, 0, .99)
        desired = 2
        self.assertEqual(idx, desired)
    
    def test_four_point(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 4.0]])
        idx, _, _ = evaluation.get_neighbourhood_points(points, 3, 0, .9)
        desired = 2
        self.assertEqual(idx, desired)