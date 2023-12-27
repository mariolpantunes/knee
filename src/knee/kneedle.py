# coding: utf-8

'''
The following module provides knee detection method
based on Kneedle algorithm.
'''

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''

import math
import enum
import logging
import numpy as np
import uts.ema as ema
import uts.peak_detection as pd
import knee.multi_knee as mk
import knee.linear_fit as lf


logger = logging.getLogger(__name__)


class Direction(enum.Enum):
    """
    Enum data type that represents the direction of a concavity.
    """
    Increasing = 'increasing'
    Decreasing = 'decreasing'

    def __str__(self):
        return self.value


class Concavity(enum.Enum):
    """
    Enum data type that represents the rotation of a concavity.
    """
    Counterclockwise = 'counter-clockwise'
    Clockwise = 'clockwise'

    def __str__(self):
        return self.value


class PeakDetection(enum.Enum):
    """
    Enum data type that identifies the peak selection algorithm.
    """
    Kneedle = 'Kneedle'
    ZScore = 'ZScore'
    Significant = 'Significant'
    All = 'All'

    def __str__(self):
        return self.value


def differences(points: np.ndarray, cd: Direction, cc: Concavity) -> np.ndarray:
    """
    Computes the differences from the y axis.

    These differences represent a rotation within the original algorithm.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        cd (Direction): direction of the concavity
        cc (Concavity): rotation of the concavity

    Returns:
        np.ndarray: the points array with the differences
    """

    rv = np.empty(points.shape)

    if cd is Direction.Decreasing and cc is Concavity.Clockwise:
        for i in range(0, len(points)):
            rv[i][0] = points[i][0]
            rv[i][1] = points[i][0] + points[i][1]  # x + y
    elif cd is Direction.Decreasing and cc is Concavity.Counterclockwise:
        for i in range(0, len(points)):
            rv[i][0] = points[i][0]
            rv[i][1] = 1.0 - (points[i][0] + points[i][1])  # 1.0 - (x + y)
    elif cd is Direction.Increasing and cc is Concavity.Clockwise:
        for i in range(0, len(points)):
            rv[i][0] = points[i][0]
            rv[i][1] = points[i][1] - points[i][0]  # y - x
    else:
        for i in range(0, len(points)):
            rv[i][0] = points[i][0]
            rv[i][1] = math.fabs(points[i][1] - points[i][0])  # abs(y - x)

    return rv


def _knee(points: np.ndarray, t: float, cd: Direction, cc: Concavity) -> int:
    """
    Returns the index of the knee point based on the Kneedle method.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t (float): tau of the side window used to smooth the curve
        cd (Direction): direction of the concavity
        cc (Concavity): rotation of the concavity

    Returns:
        int: the index of the knee point
    """

    Ds = ema.linear(points, t)
    pmin = Ds.min(axis=0)
    pmax = Ds.max(axis=0)
    diff = pmax - pmin
    diff[diff == 0] = 1.0

    Dn = (Ds - pmin)/diff 

    Dd = differences(Dn, cd, cc)

    peaks = pd.all_peaks(Dd)
    
    idx = pd.highest_peak(Dd, peaks)
    if idx == -1:
        return None
    else:
        return idx


def _knees(points: np.ndarray, t: float, cd: Direction, cc: Concavity, sensitivity:float=1.0, p:PeakDetection=PeakDetection.Kneedle, debug:bool=False) -> np.ndarray:
    """
    Returns the index of the knees point based on the Kneedle method.

    This implementation uses an heuristic to automatically define
    the direction and rotation of the concavity.

    Furthermore, it support three different methods to select the 
    relevant knees:
    1. Kneedle    : classical algorithm
    2. Significant: significant knee peak detection
    3. ZScore     : significant knee peak detection based on zscore

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t (float): tau of the sliding window used to smooth the curve
        cd (Direction): direction of the concavity
        cc (Concavity): rotation of the concavity
        sensitivity (float): controls the sensitivity of the peak detection (default 1.0)
        p (PeakDetection): selects the peak detection method (default PeakDetection.Kneedle)
        debug (bool): debug flag; when True the algorithm returns more information

    Returns:
        np.ndarray: the indexes of the knee points
    """
    Ds = ema.linear(points, t)

    pmin = Ds.min(axis=0)
    pmax = Ds.max(axis=0)
    Dn = (Ds - pmin)/(pmax - pmin)

    Dd = differences(Dn, cd, cc)

    knees = []

    peaks_idx = pd.all_peaks(Dd)

    if p is PeakDetection.Kneedle:
        knees = pd.kneedle_peak_detection(Dd, peaks_idx, sensitivity)
    elif p is PeakDetection.Significant:
        knees = pd.significant_peaks(Dd, peaks_idx, sensitivity)
    elif p is PeakDetection.ZScore:
        knees = pd.significant_zscore_peaks(Dd, peaks_idx, sensitivity)
    else:
        knees = peaks_idx

    if debug is True:
        return {'knees': knees, 'dd': Dd, 'peaks': pd.all_peaks(Dd)}
    else:
        return knees


def knees(points: np.ndarray,  t: float = 1.0, sensitivity: float = 1.0, p: PeakDetection = PeakDetection.Kneedle) -> np.ndarray:
    """
    Returns the index of the knees point based on the Kneedle method.

    This implementation uses an heuristic to automatically define
    the direction and rotation of the concavity.

    Furthermore, it support three different methods to select the 
    relevant knees:
    1. Kneedle    : classical algorithm
    2. Significant: significant knee peak detection
    3. ZScore     : significant knee peak detection based on zscore

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t (float): tau of the side window used to smooth the curve
        sensitivity (float): controls the sensitivity of the peak detection
        p (PeakDetection): selects the peak detection method

    Returns:
        np.ndarray: the indexes of the knee points
    """
    _, m = lf.linear_fit_points(points)

    if m > 0.0:
        cd = Direction.Increasing
    else:
        cd = Direction.Decreasing

    knees_1= _knees(points, t, cd, Concavity.Counterclockwise, sensitivity, p)
    knees_2 = _knees(points, t, cd, Concavity.Clockwise, sensitivity, p)

    knees_idx = np.concatenate((knees_1, knees_2))
    # np.concatenate generates float array when one is empty (see https://github.com/numpy/numpy/issues/8878)
    knees_idx = knees_idx.astype(int)
    knees_idx = np.unique(knees_idx)
    knees_idx.sort()

    return knees_idx


def knee(points: np.ndarray, t: float = 1.0) -> int:
    """
    Returns the index of the knee point based on the Kneedle method.

    This implementation uses an heuristic to automatically define
    the direction and rotation of the concavity.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t (float): tau of the side window used to smooth the curve

    Returns:
        int: the index of the knee point
    """
    b, m = lf.linear_fit_points(points)

    if m > 0.0:
        cd = Direction.Increasing
    else:
        cd = Direction.Decreasing

    y = points[:, 1]
    yhat = np.empty(len(points))
    for i in range(0, len(points)):
        yhat[i] = points[i][0]*m+b

    vote = np.sum(y - yhat)

    if cd is Direction.Increasing and vote > 0:
        cc = Concavity.Clockwise
    elif cd is Direction.Increasing and vote <= 0:
        cc = Concavity.Counterclockwise
    elif cd is Direction.Decreasing and vote > 0:
        cc = Concavity.Clockwise
    else:
        cc = Concavity.Counterclockwise

    return _knee(points, t, cd, cc)


def multi_knee(points, t1=0.01, t2=3):
    """
    Recursive knee point detection based on Kneedle.

    It returns the knee points on the curve.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t1 (float): coefficient of determination threshold (default 0.01)
        t2 (int): number of points threshold (default 3)

    Returns:
        np.ndarray: knee points on the curve

    """
    knees = mk.multi_knee(knee, points, t1, t2)
    return knees
