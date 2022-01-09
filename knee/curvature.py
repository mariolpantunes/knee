# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
import numpy as np
import uts.gradient as grad
import knee.multi_knee as mk


logger = logging.getLogger(__name__)


def knee(points: np.ndarray) -> int:
    """
    Returns the index of the knee point based on the curvature equations:
    $$
    k = \\frac{|f''(x)|}{(1+[f'(2)]^2)^{\\frac{3}{2}}}
    $$

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        int: the index of the knee point
    """

    x = points[:, 0]
    y = points[:, 1]

    gradient1 = grad.cfd(x, y)
    gradient2 = grad.csd(x, y)

    curvature = np.absolute(gradient2) / ((1.0 + gradient1**2.0)**(1.5))
    # prevents the selection of the first point
    #idx = np.argmax(curvature[0:-1]) 
    idx = np.argmax(curvature[1:-1]) + 1
    return idx


def multi_knee(points: np.ndarray, t1: float = 0.01, t2: int = 3) -> np.ndarray:
    """
    Recursive knee point detection based on the curvature equations.

    It returns the knee points on the curve.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t1 (float): coefficient of determination threshold (default 0.01)
        t2 (int): number of points threshold (default 3)

    Returns:
        np.ndarray: knee points on the curve
    """
    return mk.multi_knee(knee, points, t1, t2)
