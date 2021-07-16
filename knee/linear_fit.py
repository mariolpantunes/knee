# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import enum
import logging
import numpy as np


logger = logging.getLogger(__name__)


class R2(enum.Enum):
    """
    Enum that defines the types of coefficient of determination
    """
    adjusted = 'adjusted'
    classic = 'classic'

    def __str__(self):
        return self.value


def linear_fit_points(points: np.ndarray) -> tuple:
    """Computes the linear fit for the points.

    This methods approximates the linear fit using only the
    first and last points in a curve.

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        tuple: (b, m)
    """
    x = points[:, 0]
    y = points[:, 1]
    return linear_fit(x, y)


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple:
    """Computes the linear fit for the points.

    This methods approximates the linear fit using only the
    first and last points in a curve.

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates

    Returns:
        tuple: (b, m)
    """
    m = (y[0] - y[-1])/(x[0] - x[-1])
    b = y[0] - (m*x[0])
    return (b, m)


def linear_transform(x: np.ndarray, coef: tuple) -> np.ndarray:
    """Computes the (x,y) points for an x array and the given coefficients.

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        coef (tuple): the coefficients from the linear fit

    Returns:
        tuple: (b, m)
    """
    b, m = coef
    y_hat = x * m + b
    return y_hat


def linear_r2_points(points: np.ndarray, coef: tuple, r2: R2 = R2.classic) -> float:
    """Computes the coefficient of determination (R2).

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        coef (tuple): the coefficients from the linear fit
        r2 (R2): select the type of coefficient of determination

    Returns:
        float: coefficient of determination (R2)
    """
    x = points[:, 0]
    y = points[:, 1]
    return linear_r2(x, y, coef, r2)


def linear_r2(x: np.ndarray, y: np.ndarray, coef: tuple, r2: R2 = R2.classic) -> float:
    """Computes the coefficient of determination (R2).

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates
        coef (tuple): the coefficients from the linear fit
        r2 (R2): select the type of coefficient of determination

    Returns:
        float: coefficient of determination (R2)
    """
    y_hat = linear_transform(x, coef)
    y_mean = np.mean(y)
    rss = np.sum((y-y_hat)**2)
    tss = np.sum((y-y_mean)**2)
    rv = 0.0

    if tss == 0:
        rv = 1.0 - rss
    else:
        rv = 1.0 - (rss/tss)

    if r2 is R2.adjusted:
        rv = 1.0 - (1.0 - rv)*((len(x)-1)/(len(x)-2))

    return rv


def linear_residuals(x: np.ndarray, y: np.ndarray, coef: tuple) -> float:
    """Computes the residual error of the linear fit.

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates
        coef (tuple): the coefficients from the linear fit
        
    Returns:
        float: residual error of the linear fit
    """
    y_hat = linear_transform(x, coef)
    rss = np.sum((y-y_hat)**2)
    return rss


def r2_points(points: np.ndarray, t: R2 = R2.classic) -> float:
    """Computes the coefficient of determination (R2).

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t (R2): select the type of coefficient of determination

    Returns:
        float: coefficient of determination (R2)
    """
    if len(points) <= 2:
        return 1.0
    else:
        x = points[:, 0]
        y = points[:, 1]
        return r2(x, y, r2)


def r2(x: np.ndarray, y: np.ndarray, t: R2 = R2.classic) -> float:
    """Computes the coefficient of determination (R2).

    Computes the best fit (and not the fast point fit)
    and computes the corresponding R2.

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates
        t (R2): select the type of coefficient of determination

    Returns:
        float: coefficient of determination (R2)
    """
    rv = 0.0
    if len(x) <= 2:
        rv = 1.0
    else:
        rv = (np.corrcoef(x, y)[0, 1])**2.0
    
    if t is R2.adjusted:
        rv = 1.0 - (1-rv)*((len(x)-1)/(len(x)-2))

    return rv 
