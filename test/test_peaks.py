import unittest
from unittest import result
import numpy as np
import numpy.testing as npt

from knee import peak_detection


class Test_Peaks(unittest.TestCase):
    def test_all_peaks(self):
        points = np.array([[1.0, 5.0], [2.0, -2.0], [3.0, 3.0],
        [4.0, 1.0], [5.0, 1.0], [6.0, 4.0], [7.0, 4.0], 
        [8.0, -3.0], [9.0, -3.0], [10.0, -5.0], [11.0, 6.0], 
        [12.0, 3.0], [13.0, 0.0], [14.0, 4.0], [15.0, -2.0],
        [16.0, 7.0], [17.0, -7.0], [18.0, -3.0], [19.0, 0.0], [20.0, 5.0]])
        result = peak_detection.all_peaks(points)
        desired = np.array([False,False,True,False,False,False,False,False,False,False,True,False,False,True,False,True,False,False,False,False])
        npt.assert_almost_equal(result, desired, decimal=2)
    
    def test_significant_peaks(self):
        points = np.array([[1.0, 5.0], [2.0, -2.0], [3.0, 3.0],
        [4.0, 1.0], [5.0, 1.0], [6.0, 4.0], [7.0, 4.0], 
        [8.0, -3.0], [9.0, -3.0], [10.0, -5.0], [11.0, 6.0], 
        [12.0, 3.0], [13.0, 0.0], [14.0, 4.0], [15.0, -2.0],
        [16.0, 7.0], [17.0, -7.0], [18.0, -3.0], [19.0, 0.0], [20.0, 5.0]])
        peaks_idx = np.array([False,False,True,False,False,False,False,False,False,False,True,False,False,True,False,True,False,False,False,False])
        result = peak_detection.significant_peaks(points, peaks_idx)
        desired = np.array([False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False])
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_significant_zscore_peaks(self):
        points = np.array([[1.0, 5.0], [2.0, -2.0], [3.0, 3.0],
        [4.0, 1.0], [5.0, 1.0], [6.0, 4.0], [7.0, 4.0], 
        [8.0, -3.0], [9.0, -3.0], [10.0, -5.0], [11.0, 6.0], 
        [12.0, 3.0], [13.0, 0.0], [14.0, 4.0], [15.0, -2.0],
        [16.0, 7.0], [17.0, -7.0], [18.0, -3.0], [19.0, 0.0], [20.0, 5.0]])
        peaks_idx = np.array([False,False,True,False,False,False,False,False,False,False,True,False,False,True,False,True,False,False,False,False])
        result = peak_detection.significant_zscore_peaks(points, peaks_idx)
        desired = np.array([False,False,True,False,False,False,False,False,False,False,True,False,False,True,False,True,False,False,False,False])
        npt.assert_almost_equal(result, desired, decimal=2)


if __name__ == '__main__':
    unittest.main()