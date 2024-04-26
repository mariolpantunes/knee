# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '1.0'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''

import unittest
import numpy as np
import kneeliverse.zmethod as zmethod


class TestZ_Method(unittest.TestCase):
    def test_multi_knee(self):
        x = np.array([0,1,2,3,4,5,6,7,8,9])
        y = np.array([1,0.5,0.333333333,0.25,0.2,0.2,0.1,0.06666666667,0.05,0.04])
        points = np.stack((x, y), axis=1)
        result = zmethod.knees(points)
        desired = np.array([0, 1, 2, 3, 4, 6, 9])
        np.testing.assert_array_equal(result, desired)


if __name__ == '__main__':
    unittest.main()