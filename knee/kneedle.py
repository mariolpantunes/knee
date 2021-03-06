# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


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

    def __str__(self):
        return self.value


def differences(points: np.ndarray, cd: Direction, cc: Concavity) -> np.ndarray:
    """Computes the differences from the y axis.

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


def single_knee(points: np.ndarray, t: float, cd: Direction, cc: Concavity) -> int:
    """Returns the index of the knee point based on the Kneedle method.

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
    Dn = (Ds - pmin)/(pmax - pmin)
    Dd = differences(Dn, cd, cc)
    peaks = pd.all_peaks(Dd)
    idx = pd.highest_peak(points, peaks)
    if idx == -1:
        return None
    else:
        return idx


def knees(points: np.ndarray, t: float, sensitivity: float, cd: Direction, cc: Concavity, p: PeakDetection) -> np.ndarray:
    """Returns the index of the knees point based on the Kneedle method.

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
        cd (Direction): direction of the concavity
        cc (Concavity): rotation of the concavity
        p (PeakDetection): selects the peak detection method

    Returns:
        np.ndarray: the indexes of the knee points
    """
    Ds = ema.linear(points, t)

    pmin = Ds.min(axis=0)
    pmax = Ds.max(axis=0)
    Dn = (Ds - pmin)/(pmax - pmin)

    Dd = differences(Dn, cd, cc)

    knees = []

    if p is PeakDetection.Kneedle:
        idx = []
        lmxThresholds = []
        detectKneeForLastLmx = False
        for i in range(1, len(Dd)-1):
            y0 = Dd[i-1][1]
            y = Dd[i][1]
            y1 = Dd[i+1][1]

            if y0 < y and y > y1:
                idx.append(i)
                tlmx = y - sensitivity / (len(Dd) - 1)
                lmxThresholds.append(tlmx)
                detectKneeForLastLmx = True

            if detectKneeForLastLmx:
                if y1 < lmxThresholds[-1]:
                    knees.append(idx[-1])
                    detectKneeForLastLmx = False
        knees = np.array(knees)

    elif p is PeakDetection.Significant:
        peaks_idx = pd.all_peaks(Dd)
        knees = pd.significant_peaks(Dd, peaks_idx, sensitivity)

    elif p is PeakDetection.ZScore:
        peaks_idx = pd.all_peaks(Dd)
        knees = pd.significant_zscore_peaks(Dd, peaks_idx, sensitivity)

    return knees


def auto_knees(points: np.ndarray,  t: float = 1.0, sensitivity: float = 1.0, p: PeakDetection = PeakDetection.Kneedle) -> np.ndarray:
    """Returns the index of the knees point based on the Kneedle method.

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

    knees_1 = knees(points, sensitivity, t, cd, Concavity.Counterclockwise, p)
    knees_2 = knees(points, sensitivity, t, cd, Concavity.Clockwise, p)

    knees_idx = np.concatenate((knees_1, knees_2))
    knees_idx = np.unique(knees_idx)
    knees_idx.sort()

    return knees_idx


def auto_knee(points: np.ndarray, t: float = 1.0) -> int:
    """Returns the index of the knee point based on the Kneedle method.

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

    return single_knee(points, t, cd, cc)


def multi_knee(points, t1=0.99, t2=3):
    """
    Recursive knee point detection based on Kneedle.

    It returns the knee points on the curve.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t1 (float): coefficient of determination threshold (default 0.99)
        t2 (int): number of points threshold (default 3)

    Returns:
        np.ndarray: knee points on the curve

    """
    return mk.multi_knee(auto_knee, points, t1, t2)
