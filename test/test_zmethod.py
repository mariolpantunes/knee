
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

from kneeliverse import zmethod


class TestZ_Method(unittest.TestCase):
    def test_multi_knee(self):
        x = np.array([0,1,2,3,4,5,6,7,8,9])
        y = np.array([1,0.5,0.333333333,0.25,0.2,0.2,0.1,0.06666666667,0.05,0.04])
        points = np.stack((x, y), axis=1)
        result = zmethod.knees(points)
        desired = np.array([0, 1, 2, 3, 4, 6, 9])
        np.testing.assert_array_equal(result, desired)


class TestMapIndex(unittest.TestCase):
    """Maps knee x VALUES back to their indexes in the original x array.

    Takes a 1-D x array, not the (x, y) point matrix - passing points raises
    "object too deep for desired array", which is easy to hit because most
    of the module takes points.
    """

    def setUp(self):
        self.x = np.arange(20, dtype=float)

    def test_it_recovers_the_indexes(self):
        np.testing.assert_array_equal(zmethod.map_index(self.x, self.x[[3, 7, 12]]),
                                      np.array([3, 7, 12]))

    def test_it_works_on_unsorted_input(self):
        x = np.array([5.0, 1.0, 9.0, 3.0])
        np.testing.assert_array_equal(zmethod.map_index(x, np.array([9.0, 1.0])),
                                      np.array([2, 1]))

    def test_an_empty_query_gives_an_empty_result(self):
        self.assertEqual(len(zmethod.map_index(self.x, np.array([]))), 0)


class TestZMethodDetectors(unittest.TestCase):
    def setUp(self):
        y = np.concatenate([np.linspace(1.0, 0.3, 8), np.full(12, 0.3)])
        self.points = np.column_stack((np.arange(len(y), dtype=float), y))

    def test_knees_returns_valid_indexes(self):
        knees = zmethod.knees(self.points)
        self.assertTrue(np.all(np.asarray(knees) >= 0))
        self.assertTrue(np.all(np.asarray(knees) < len(self.points)))

    def test_knees2_returns_valid_indexes(self):
        knees = np.asarray(zmethod.knees2(self.points))
        self.assertTrue(np.all(knees >= 0))
        self.assertTrue(np.all(knees < len(self.points)))

    def test_knees2_accepts_every_outlier_method(self):
        for out in zmethod.Outlier:
            with self.subTest(outlier=out.value):
                self.assertIsNotNone(zmethod.knees2(self.points, out=out))

    def test_getpoints_returns_indexes_by_default(self):
        result = np.asarray(zmethod.getPoints(self.points))
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result < len(self.points)))

    def test_getpoints_with_plot_returns_the_working_data(self):
        # The `plot` flag changes the return TYPE - a (dict, array) pair
        # instead of an index array - which is worth pinning explicitly.
        result = zmethod.getPoints(self.points, plot=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], dict)


if __name__ == '__main__':
    unittest.main()