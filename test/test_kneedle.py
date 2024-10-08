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
import kneeliverse.kneedle as kneedle


class TestKneedle(unittest.TestCase):
    def test_get_knee_naive(self):
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ])
        y = np.array([1, 0.5, 0.333333333, 0.25, 0.2, 0.166666667, 0.142857143, 0.125, 0.111111111, 0.1])
        points = np.stack((x, y), axis=1)
        result = kneedle.knee(points)
        desired = 4
        self.assertEqual(result, desired)

    def test_multi_knee(self):
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ])
        y = np.array([1, 0.5, 0.333333333, 0.25, 0.2, 0.2, 0.1, 0.06666666667, 0.05, 0.04])
        points = np.stack((x, y), axis=1)
        result = kneedle.multi_knee(points)
        desired = np.array([2, 3, 8])
        np.testing.assert_array_equal(result, desired)

    def test_knees_classic(self):
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ])
        y = np.array([1, 0.5, 0.333333333, 0.25, 0.2,
                     0.2, 0.1, 0.06666666667, 0.05, 0.04])
        points = np.stack((x, y), axis=1)
        result = kneedle.knees(points)
        desired = np.array([3])
        np.testing.assert_array_equal(result, desired)

    def test_knees_significant(self):
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ])
        y = np.array([1, 0.5, 0.333333333, 0.25, 0.2,
                     0.2, 0.1, 0.06666666667, 0.05, 0.04])
        points = np.stack((x, y), axis=1)
        result = kneedle.knees(points, sensitivity=1.0, 
        p=kneedle.PeakDetection.Significant)
        desired = np.array([3])
        np.testing.assert_array_equal(result, desired)

    def test_knees_zscore(self):
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ])
        y = np.array([1, 0.5, 0.333333333, 0.25, 0.2,
                     0.2, 0.1, 0.06666666667, 0.05, 0.04])
        points = np.stack((x, y), axis=1)
        result = kneedle.knees(points, sensitivity=1.0, p=kneedle.PeakDetection.ZScore)
        desired = np.array([3])
        np.testing.assert_array_equal(result, desired)

    def test_knee_noisy(self):
        x = np.array([2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
        15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,
        41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,
        54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,
        67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
        80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,
        93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104])
        y = np.array([1.46642427e+16, 1.46360638e+16, 1.46109304e+16, 1.40241078e+16,
        1.39973697e+16, 1.37339563e+16, 1.37080843e+16, 1.36822286e+16,
        1.09277805e+16, 7.25824884e+15, 7.23051996e+15, 7.21416014e+15,
        7.20647085e+15, 7.06249332e+15, 7.03655202e+15, 7.01450636e+15,
        7.01298336e+15, 6.88391133e+15, 5.66408261e+15, 5.63855379e+15,
        5.62626291e+15, 5.61667644e+15, 5.61644576e+15, 4.81521120e+15,
        4.80992419e+15, 2.89667429e+15, 2.89093407e+15, 2.88275725e+15,
        2.67631575e+15, 2.64999808e+15, 2.58947181e+15, 2.58395401e+15,
        1.64421282e+15, 1.20427073e+15, 1.19933334e+15, 1.19883050e+15,
        1.19589100e+15, 1.17739756e+15, 1.17511666e+15, 1.14576400e+15,
        1.14246443e+15, 1.13963244e+15, 1.13232824e+15, 1.04073860e+15,
        1.03564211e+15, 1.03192774e+15, 1.01086507e+15, 1.00884228e+15,
        9.44326412e+14, 8.02861515e+14, 7.93601804e+14, 7.93275232e+14,
        7.62876758e+14, 7.62761206e+14, 7.53740501e+14, 7.11238319e+14,
        5.08782908e+14, 5.05262329e+14, 4.98180470e+14, 4.91607239e+14,
        4.87506346e+14, 4.52247920e+14, 4.37373930e+14, 4.35770343e+14,
        4.31725646e+14, 4.28800040e+14, 4.26539621e+14, 4.22103050e+14,
        3.46869510e+14, 3.43448655e+14, 3.37483722e+14, 3.36001659e+14,
        3.16352332e+14, 3.09390335e+14, 3.08924254e+14, 2.93465268e+14,
        1.82387049e+14, 1.79325963e+14, 1.76244921e+14, 6.56817319e+13,
        6.40335339e+13, 6.22978474e+13, 5.57563164e+13, 5.55395472e+13,
        5.50342478e+13, 5.15280272e+13, 4.93742096e+13, 4.91594202e+13,
        4.79684814e+13, 4.77723074e+13, 4.04812741e+13, 3.01677265e+13,
        2.79967578e+13, 2.13732819e+13, 2.02269664e+13, 1.97349492e+13,
        1.53887156e+13, 4.29217421e+12, 2.73160671e+12, 2.47734172e+12,
        4.47778860e+11, 1.02093688e+11, 4.39657822e+10])
        points = np.stack((x, y), axis=1)
        result = kneedle.knee(points, t=1.0)
        desired = 34
        self.assertEqual(result, desired)


if __name__ == '__main__':
    unittest.main()
