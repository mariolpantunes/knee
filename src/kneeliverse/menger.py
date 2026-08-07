# coding: utf-8

'''
The following module provides knee detection method
based on menger curvature.
'''

__author__ = 'Mário Antunes'
__version__ = '1.0'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''

import math
import logging
import numpy as np
import kneeliverse.knee_ranking as kr
import kneeliverse.multi_knee as mk


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
    Returns the index of the knee point based on the menger curvature.

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
    # argmax_tol, not np.argmax - see the note in curvature.knee. Menger
    # curvature is a ratio of triangle areas, so three collinear or equally
    # bent point-triples give values that agree mathematically and differ in
    # their last bits; ties resolve to the leftmost point.
    return kr.argmax_tol(curvature)


def multi_knee(points: np.ndarray, t1: float = 0.001, t2: int = 4) -> np.ndarray:
    """Recursive knee point detection based on the menger curvature.

    It returns the knee points on the curve.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t1 (float): coefficient of determination threshold (default 0.01)
        t2 (int): number of points threshold (default 4)

    Returns:
        np.ndarray: knee points on the curve
    """
    return mk.multi_knee(knee, points, t1, t2)
