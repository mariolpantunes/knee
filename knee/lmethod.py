# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import logging
import numpy as np
import knee.linear_fit as lf 
import knee.multi_knee as mk


from enum import Enum


logger = logging.getLogger(__name__)


class Fit(Enum):
    """
    Enum that defines the types of linear fitting
    """
    best_fit = 'bestfit'
    point_fit = 'pointfit'

    def __str__(self):
        return self.value


def get_knee(x: np.ndarray, y: np.ndarray, fit=Fit.point_fit) -> int:
    """
    Returns the index of the knee point based on menger curvature.

    This method uses the iterative refinement.

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates
        fit (Fit): select between point fit and best fit

    Returns:
        int: the index of the knee point
    """

    index = 2
    length = x[-1] - x[0]
    left_length = x[index] - x[0]
    right_length = x[-1] - x[index]

    if fit is Fit.best_fit:
        coef_left, r_left, *_  = np.polyfit(x[0:index+1], y[0:index+1], 1, full=True)
        coef_right, r_rigth, *_ = np.polyfit(x[index:], y[index:], 1, full=True)
        error = r_left[0]*(left_length/length) + r_rigth[0]*(right_length/length)
        #error = (r_left[0] + r_rigth[0]) / 2.0
    else:
        coef_left = lf.linear_fit(x[0:index+1], y[0:index+1])
        coef_right = lf.linear_fit(x[index:], y[index:])
        r_left = lf.linear_residuals(x[0:index+1], y[0:index+1], coef_left)
        r_rigth = lf.linear_residuals(x[index:], y[index:], coef_right)
        error = r_left*(left_length/length) + r_rigth*(right_length/length)
        #error = (r_left + r_rigth) / 2.0
    
    #logger.info("Error(%s) = %s", index, error)

    for i in range(index+1, len(x)-2):
        left_length = x[i] - x[0]
        right_length = x[-1] - x[i]

        if fit is Fit.best_fit:
            i_coef_left, r_left, *_  = np.polyfit(x[0:i+1], y[0:i+1], 1, full=True)
            i_coef_right, r_rigth, *_ = np.polyfit(x[i:], y[i:], 1, full=True)
            current_error = r_left[0]*(left_length/length) + r_rigth[0]*(right_length/length)
            #current_error = (r_left[0] + r_rigth[0]) / 2.0
        else:
            i_coef_left = lf.linear_fit(x[0:i+1], y[0:i+1])
            i_coef_right = lf.linear_fit(x[i:], y[i:])
            r_left = lf.linear_residuals(x[0:i+1], y[0:i+1], i_coef_left)
            r_rigth = lf.linear_residuals(x[i:], y[i:], i_coef_right)
            current_error = r_left*(left_length/length) + r_rigth*(right_length/length)
            #current_error = (r_left + r_rigth) / 2.0
        
        #logger.info("Error(%s) = %s", i, error)

        if current_error < error:
            error = current_error
            index = i
            coef_left = i_coef_left
            coef_right = i_coef_right

    return (index, coef_left, coef_right)


def knee(points:np.ndarray, fit:Fit=Fit.point_fit) -> int:
    """
    Returns the index of the knee point based on the L-method.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        fit (Fit): select between point fit and best fit

    Returns:
        int: the index of the knee point
    """

    x = points[:,0]
    y = points[:,1]

    last_knee = -1
    cutoff  = current_knee = len(x)
    #logger.info('Knee [%s, %s, %s]', cutoff, last_knee, current_knee)

    done = False
    while current_knee != last_knee and not done:
        last_knee = current_knee
        current_knee, _, _ = get_knee(x[0:cutoff+1], y[0:cutoff+1], fit)
        cutoff = int((current_knee + last_knee)/2)
        if cutoff < 10:
            done = True
        
    return current_knee


def multi_knee(points: np.ndarray, t1: float = 0.01, t2: int = 4) -> np.ndarray:
    """Recursive knee point detection based on the L-method.

    It returns the knee points on the curve.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t1 (float): coefficient of determination threshold (default 0.01)
        t2 (int): number of points threshold (default 3)

    Returns:
        np.ndarray: knee points on the curve
    """
    return mk.multi_knee(knee, points, t1, t2)