# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import logging
import numpy as np
import uts.gradient as grad
import uts.thresholding as thresh
import knee.multi_knee as mk


logger = logging.getLogger(__name__)


def get_knee(x: np.ndarray, y: np.ndarray) -> int:
    """Return the index of the knee point based on the DFDT method.

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates

    Returns:
        int: the index of the knee point
    """
    gradient = grad.cfd(x, y)
    return get_knee_gradient(gradient)


def get_knee_gradient(gradient: np.ndarray) -> int:
    """Return the index of the knee point based on the DFDT method.

    Args:
        gradient (np.ndarray): the first order gradient of the trace points

    Returns:
        int: the index of the knee point
    """
    t = thresh.isodata(gradient)
    diff = np.absolute(gradient - t)
    knee = np.argmin(diff[1:-1]) + 1
    return knee


def knee(points: np.ndarray) -> int:
    """Returns the index of the knee point based on the DFDT method.

    It uses the iterative refinement  method.

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        int: the index of the knee point
    """
    x = points[:, 0]
    y = points[:, 1]

    gradient = grad.cfd(x, y)

    knee = cutoff = 0
    last_knee = -1

    while last_knee < knee and (len(x)-cutoff) > 2:
        last_knee = knee
        knee = get_knee_gradient(gradient[cutoff:]) + cutoff
        cutoff = int(math.ceil(knee/2.0))

    return knee


def multi_knee(points: np.ndarray, t1: float = 0.99, t2: int = 3) -> np.ndarray:
    """Recursive knee point detection based on DFDT.

    It returns the knee points on the curve.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t1 (float): coefficient of determination threshold (default 0.99)
        t2 (int): number of points threshold (default 3)

    Returns:
        np.ndarray: The knee points on the curve
    """
    return mk.multi_knee(knee, points, t1, t2)
