# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import logging
import numpy as np
from knee.linear_fit import linear_fit, linear_r2
from uts import ema
import knee.multi_knee as mk


logger = logging.getLogger(__name__)


def menger_curvature(f:np.ndarray, g:np.ndarray, h:np.ndarray) -> float:
    """
    Computes the menger curvature based on three points.

    Args:
        f (np.ndarray): first point
        g (np.ndarray): second point
        h (np.ndarray): third point

    Returns:
        float: menger curvature
    """
    x1 = f[0]
    y1 = f[1]
    x2 = g[0]
    y2 = g[1]
    x3 = h[0]
    y3 = h[1]

    nom = 2.0 * math.fabs((x2-x1)*(y3-y2))-((y2-y1)*(x3-x2))
    temp = math.fabs((x2-x1)**2.0 + (y2-y1)**2.0)*math.fabs((x3-x2)
    ** 2.0 + (y3-y2)**2.0) * math.fabs((x1-x3)**2.0 + (y1-y3)**2.0)
    dem = math.sqrt(temp)

    return nom/dem


def knee(points: np.ndarray) -> int:
    """
    Returns the index of the knee point based on menger curvature.

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        int: the index of the knee point
    """

    curvature = [0]

    for i in range(1, len(points)-1):
        f = points[i]
        g = points[i-1]
        h = points[i+1]

        curvature.append(menger_curvature(f, g, h))

    curvature.append(0)
    curvature = np.array(curvature)
    return np.argmax(curvature)


def multi_knee(points: np.ndarray, t1: float = 0.99, t2: int = 4) -> np.ndarray:
    """Recursive knee point detection based on the menger curvature.

    It returns the knee points on the curve.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t1 (float): coefficient of determination threshold (default 0.99)
        t2 (int): number of points threshold (default 3)

    Returns:
        np.ndarray: knee points on the curve
    """
    return mk.multi_knee(knee, points, t1, t2)
