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


class Cost(Enum):
    """
    Enum that defines the cost used in the L-method
    
    RMSE was originaly used by the original authors.
    RSS can be used to speedup the computation 
    (it has similar results in most practical cases).
    """
    rss = 'rss'
    rmse = 'rmse'

    def __str__(self):
        return self.value


def compute_error(x: np.ndarray, y: np.ndarray, index:int, length:int, fit=Fit.point_fit, cost=Cost.rmse):
    """
    Returns the fitting error that is minimized by the L-method algorithm.

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates
        index (int): the index where the fitting line is divided
        lenght (int): the lenght of the points considered
        fit (Fit): select between point fit and best fit
        cost (Cost): use either RMSE (original work) or RSS (faster implementation)
    
    Returns:
        float: the fitting cost
    """
    
    left_length = x[index] - x[0]
    right_length = x[-1] - x[index]

    left_ratio = left_length/length
    right_ratio = right_length/length

    if fit is Fit.best_fit:
        coef_left, r_left, *_  = np.polyfit(x[0:index+1], y[0:index+1], 1, full=True)
        coef_right, r_rigth, *_ = np.polyfit(x[index:], y[index:], 1, full=True)
        #error = r_left[0]*(left_length/length) + r_rigth[0]*(right_length/length)
        r_left = r_left[0]
        r_rigth = r_rigth[0]
    else:
        coef_left = lf.linear_fit(x[0:index+1], y[0:index+1])
        coef_right = lf.linear_fit(x[index:], y[index:])
        r_left = lf.linear_residuals(x[0:index+1], y[0:index+1], coef_left)
        r_rigth = lf.linear_residuals(x[index:], y[index:], coef_right)
        
    if cost is Cost.rmse:
        error = left_ratio*math.sqrt(r_left*left_ratio) + right_ratio*math.sqrt(right_ratio*r_rigth)
    else:
        error = r_left*left_ratio + r_rigth*right_ratio
    
    return error, coef_left, coef_right


def get_knee(x: np.ndarray, y: np.ndarray, fit=Fit.point_fit, cost=Cost.rmse) -> int:
    """
    Returns the index of the knee point based on the L-method.

    This method does not use the iterative refinement.
    It represents a single iteration of the refinement technique.

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates
        fit (Fit): select between point fit and best fit

    Returns:
        int: the index of the knee point
    """

    index = 2
    length = x[-1] - x[0]
    error, coef_left, coef_right = compute_error(x, y, index, length, fit, cost)
    #logger.info("Error(%s) = %s", index, error)

    for i in range(index+1, len(x)-2):
        current_error, i_coef_left, i_coef_right = compute_error(x, y, i, length, fit, cost)
        #logger.info("Error(%s) = %s", i, error)

        if current_error < error:
            error = current_error
            index = i
            coef_left = i_coef_left
            coef_right = i_coef_right

    return (index, coef_left, coef_right)


def knee(points:np.ndarray, fit:Fit=Fit.point_fit, it:bool=True) -> int:
    """
    Returns the index of the knee point based on the L-method.

    This method uses the iterative refinement.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        fit (Fit): select between point fit and best fit
        it (bool): flag the usage of iterative refinement (True)

    Returns:
        int: the index of the knee point
    """

    x = points[:,0]
    y = points[:,1]

    last_knee = -1
    cutoff  = current_knee = len(x)
    done = False
    while current_knee != last_knee and not done:
        last_knee = current_knee
        current_knee, _, _ = get_knee(x[0:cutoff+1], y[0:cutoff+1], fit)
        cutoff = int((current_knee + last_knee)/2)
        if cutoff < 10 or not it:
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