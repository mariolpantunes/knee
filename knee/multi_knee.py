# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import typing
import logging
import numpy as np
from knee.linear_fit import linear_fit_points, linear_r2_points


logger = logging.getLogger(__name__)


def multi_knee(get_knee:typing.Callable, points:np.ndarray, t1:float=0.99, t2:int=3) -> np.ndarray:
    """
    Wrapper that convert a single knee point detection into a multi knee point detector.

    It uses recursion on the left and right parts of the curve after detecting the current knee.

    Args:
        get_knee (typing.Callable): method that returns a single knee point
        points (np.ndarray): numpy array with the points (x, y)
        t1 (float): the coefficient of determination used as a threshold (default 0.99)
        t2 (int): the mininum number of points used as a threshold (default 3)

    Returns:
        np.ndarray: knee points on the curve
    """
    return np.array(multi_knee_rec(get_knee, points, 0, len(points), t1, t2))


def multi_knee_rec(get_knee:typing.Callable, points:np.ndarray, left:int, right:int, t1:float, t2:int) -> np.ndarray:
    """
    Wrapper that convert a single knee point detection into a multi knee point detector.

    It uses recursion on the left and right parts of the curve after detecting the current knee.

    Args:
        get_knee (typing.Callable): method that returns a single knee point
        points (np.ndarray): numpy array with the points (x, y)
        left (int): the left limit of the current part
        right (int): the right limit of the current part
        t1 (float): the coefficient of determination used as a threshold (default 0.99)
        t2 (int): the mininum number of points used as a threshold (default 3)

    Returns:
        np.ndarray: knee points on the curve
    """
    logger.debug('[%s, %s]', left, right)
    pt = points[left:right]
    if len(pt) > t2:
        coef = linear_fit_points(pt)
        if linear_r2_points(pt, coef) < t1:
            rv = get_knee(pt)
            logger.debug('RV -> %s', rv)
            if rv is not None:
                idx = rv + left
                left_knees = multi_knee_rec(get_knee, points, left, idx+1, t1, t2)
                right_knees = multi_knee_rec(get_knee, points, idx+1, right, t1, t2)
                return left_knees + [idx] + right_knees
            else:
                return []
        else:
            return []
    else:
        return []


