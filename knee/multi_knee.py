# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import typing
import logging
import numpy as np
import knee.linear_fit as lf


logger = logging.getLogger(__name__)


def multi_knee(get_knee: typing.Callable, points: np.ndarray, t1: float = 0.01, t2: int = 3, cost: lf.Linear_Metrics = lf.Linear_Metrics.rmspe) -> np.ndarray:
    """
    Wrapper that convert a single knee point detection into a multi knee point detector.

    It uses recursion on the left and right parts of the curve after detecting the current knee.

    Args:
        get_knee (typing.Callable): method that returns a single knee point
        points (np.ndarray): numpy array with the points (x, y)
        t1 (float): the coefficient of determination used as a threshold (default 0.01)
        t2 (int): the mininum number of points used as a threshold (default 3)
        cost (lf.Linear_Metrics): the cost method used to evaluate a point set (default: lf.Linear_Metrics.rmspe)

    Returns:
        np.ndarray: knee points on the curve
    """

    stack = [(0, len(points))]
    knees = []

    while stack:
        left, right = stack.pop()
        pt = points[left:right]
        
        if len(pt) > t2:
            if len(pt) <= 2:
                if cost is lf.Linear_Metrics.rmspe:
                    r = 0.0
                else:
                    r = 1.0
            else:
                coef = lf.linear_fit_points(pt)
                if cost is lf.Linear_Metrics.rmspe:
                    r = lf.rmspe_points(pt, coef)
                else:
                    r = lf.linear_r2_points(pt, coef)

            curved = r >= t1 if cost is lf.Linear_Metrics.rmspe else r < t1

            #coef = lf.linear_fit_points(pt)
            # if lf.linear_r2_points(pt, coef) < t1:
            if curved:
                rv = get_knee(pt)
                if rv is not None:
                    idx = rv + left
                    knees.append(idx)
                    stack.append((left, idx+1))
                    stack.append((idx+1, right))
    knees.sort()
    return np.array(knees)
