# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import enum
import math
import logging
import numpy as np


logger = logging.getLogger(__name__)


class Linear_Metrics(enum.Enum):
    """
    Enum that defines the different metrics linear functions.
    These metrics are used on the `knee.rdp` and `knee.multi_knee`. 
    """
    r2 = 'r2'
    rmspe = 'rmspe'
    rpd = 'rpd'

    def __str__(self):
        return self.value


class R2(enum.Enum):
    """
    Enum that defines the types of coefficient of determination
    """
    adjusted = 'adjusted'
    classic = 'classic'

    def __str__(self):
        return self.value


def linear_fit_points(points: np.ndarray) -> tuple:
    """
    Computes the linear fit for the points.

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
    """
    Computes the linear fit for the points.

    This methods approximates the linear fit using only the
    first and last points in a curve.

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates

    Returns:
        tuple: (b, m)
    """

    d = x[0] - x[-1]
    if d != 0:
        m = (y[0] - y[-1])/(x[0] - x[-1])
        b = y[0] - (m*x[0])
        return (b, m)
    else:
        return (0, 0)


def linear_transform_points(points: np.ndarray, coef: tuple) -> np.ndarray:
    """
    Computes the y values for an x array and the given coefficients.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        coef (tuple): the coefficients from the linear fit

    Returns:
        np.ndarray: the corresponding y values
    """
    x = points[:, 0]
    return linear_transform(x, coef)


def linear_transform(x: np.ndarray, coef: tuple) -> np.ndarray:
    """
    Computes the y values for an x array and the given coefficients.

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        coef (tuple): the coefficients from the linear fit

    Returns:
        np.ndarray: the corresponding y values
    """
    b, m = coef
    y_hat = x * m + b
    return y_hat


def linear_r2_points(points: np.ndarray, coef: tuple, r2: R2 = R2.classic) -> float:
    """
    Computes the coefficient of determination (R2).

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        coef (tuple): the coefficients from the linear fit
        r2 (R2): select the type of coefficient of determination (default: R2.classic)

    Returns:
        float: coefficient of determination (R2)
    """
    x = points[:, 0]
    y = points[:, 1]
    return linear_r2(x, y, coef, r2)


def linear_r2(x: np.ndarray, y: np.ndarray, coef: tuple, r2: R2 = R2.classic) -> float:
    """
    Computes the coefficient of determination (R2).

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates
        coef (tuple): the coefficients from the linear fit
        r2 (R2): select the type of coefficient of determination (default: R2.classic)

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


def rmspe_points(points: np.ndarray, coef: tuple, eps: float = 1e-16) -> float:
    """
    Computes the Root Mean Squared Percentage Error (RMSPE).

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        coef (tuple): the coefficients from the linear fit
        eps (float): eps value to prevent division by zero (default: 1E-16)

    Returns:
        float: Root Mean Squared Percentage Error (RMSPE)
    """
    x = points[:, 0]
    y = points[:, 1]
    return rmspe(x, y, coef, eps)


def rmspe(x: np.ndarray, y: np.ndarray, coef: tuple, eps: float = 1e-16) -> float:
    """
    Computes the Root Mean Squared Percentage Error (RMSPE).

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates
        coef (tuple): the coefficients from the linear fit
        eps (float): eps value to prevent division by zero (default: 1E-16)

    Returns:
        float: Root Mean Squared Percentage Error (RMSPE)
    """
    y_hat = linear_transform(x, coef)
    rv = np.sqrt(np.mean(np.square((y - y_hat) / (y+eps))))
    return rv


def rpd_points(points: np.ndarray, coef: tuple, eps: float = 1e-16) -> float:
    """
    Computes the Relative Percentage Difference (RPD).

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        coef (tuple): the coefficients from the linear fit
        eps (float): eps value to prevent division by zero (default: 1E-16)

    Returns:
        float: Relative Percentage Difference (RPD)
    """
    x = points[:, 0]
    y = points[:, 1]
    return rpd(x, y, coef, eps)


def rpd(x: np.ndarray, y: np.ndarray, coef: tuple, eps: float = 1e-16) -> float:
    """
    Computes the Relative Percentage Difference (RPD).

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates
        coef (tuple): the coefficients from the linear fit
        eps (float): eps value to prevent division by zero (default: 1E-16)

    Returns:
        float: Relative Percentage Difference (RPD)
    """
    y_hat = linear_transform(x, coef)
    rv = np.mean(np.abs((y - y_hat) / (np.maximum(y, y_hat)+eps)))
    return rv


def rmse_points(points: np.ndarray, coef: tuple) -> float:
    """
    Computes the Root Mean Squared Error (RMSE).

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        coef (tuple): the coefficients from the linear fit

    Returns:
        float: Root Mean Squared Error (RMSE)
    """
    x = points[:, 0]
    y = points[:, 1]
    return rmse(x, y, coef)


def rmse(x: np.ndarray, y: np.ndarray, coef: tuple) -> float:
    """
    Computes the Root Mean Squared Error (RMSE).

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates
        coef (tuple): the coefficients from the linear fit

    Returns:
        float: Root Mean Squared Error (RMSE)
    """
    y_hat = linear_transform(x, coef)
    rv = np.sqrt(np.mean(np.square(y - y_hat)))
    return rv


def linear_residuals_points(points: np.ndarray, coef: tuple) -> float:
    """
    Computes the residual error of the linear fit.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        coef (tuple): the coefficients from the linear fit

    Returns:
        float: residual error of the linear fit
    """
    x = points[:, 0]
    y = points[:, 1]
    return linear_residuals(x, y, coef)


def linear_residuals(x: np.ndarray, y: np.ndarray, coef: tuple) -> float:
    """
    Computes the residual error of the linear fit.

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates
        coef (tuple): the coefficients from the linear fit

    Returns:
        float: residual error of the linear fit
    """
    y_hat = linear_transform(x, coef)
    rss = np.sum(np.square((y-y_hat)))
    return rss


def r2_points(points: np.ndarray, t: R2 = R2.classic) -> float:
    """
    Computes the coefficient of determination (R2).

    Computes the best fit (and not the fast point fit)
    and computes the corresponding R2.

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
        return r2(x, y, t)


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


def angle(coef1: tuple, coef2: tuple) -> float:
    """
    Computes the angle between two lines.

    Args:
        coef1 (tuple): the coefficients from the first line
        coef2 (tuple): the coefficients from the second line

    Returns:
        float: the angle between two lines \\( \\left[0,\\frac{\\pi}{2} \\right] \\)
    """
    _, m1 = coef1
    _, m2 = coef2
    return math.atan((m1-m2)/(1.0+m1*m2))


def shortest_distance_points(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """
    Computes the shortest distance from the points to the 
    straight line defined by the left and right point.

    Args:
        pt (np.ndarray): numpy array with the points (x, y)
        start (np.ndarray): the left point
        end (np.ndarray): the right point

    Returns:
        np.ndarray: the perpendicular distances
    """

    # TODO for you: consider implementing @Eskapp's suggestions
    if np.all(a == b):
        return np.linalg.norm(p - a, axis=1)

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(p))])

    # perpendicular distance component, as before
    # note that for the 3D case these will be vectors
    c = np.cross(p - a, d)

    # use hypot for Pythagoras to improve accuracy
    return np.hypot(h, c)


def perpendicular_distance(points: np.ndarray) -> np.ndarray:
    """
    Computes the perpendicular distance from the points to the 
    straight line defined by the first and last point.

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        np.ndarray: the perpendicular distances

    """
    return perpendicular_distance_index(points, 0, len(points) - 1)


def perpendicular_distance_index(points: np.ndarray, left: int, right: int) -> np.ndarray:
    """
    Computes the perpendicular distance from the points to the 
    straight line defined by the left and right point.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        left (int): the index of the left point
        right (int): the index of the right point

    Returns:
        np.ndarray: the perpendicular distances
    """
    return left + perpendicular_distance_points(points[left:right+1], points[left], points[right])


def perpendicular_distance_points(pt: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """
    Computes the perpendicular distance from the points to the 
    straight line defined by the left and right point.

    Args:
        pt (np.ndarray): numpy array with the points (x, y)
        start (np.ndarray): the left point
        end (np.ndarray): the right point

    Returns:
        np.ndarray: the perpendicular distances
    """
    return np.fabs(np.cross(end-start, pt-start)/np.linalg.norm(end-start))
