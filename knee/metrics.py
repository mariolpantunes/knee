# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import enum
import numpy as np


class Metrics(enum.Enum):
    """
    Enum that defines the different metrics linear functions.
    These metrics are used on the `knee.rdp` and `knee.multi_knee`. 
    """
    r2 = 'r2'
    rmspe = 'rmspe'
    rmsle = 'rmsle'
    rpd = 'rpd'
    smape = 'smape'

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


def r2(y: np.ndarray, y_hat: np.ndarray, r2: R2 = R2.classic) -> float:
    """
    Computes the coefficient of determination (R2).

    Args:
        y (np.ndarray): the real value of the points in the y axis coordinates
        y_hat (np.ndarray): the predicted value of the points in the y axis coordinates
        r2 (R2): select the type of coefficient of determination (default: R2.classic)

    Returns:
        float: coefficient of determination (R2)
    """
    y_mean = np.mean(y)
    rss = np.sum((y-y_hat)**2)
    tss = np.sum((y-y_mean)**2)
    rv = 0.0

    if tss == 0:
        rv = 1.0 - rss
    else:
        rv = 1.0 - (rss/tss)

    if r2 is R2.adjusted:
        rv = 1.0 - (1.0 - rv)*((len(y)-1)/(len(y)-2))

    return rv


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Computes the Root Mean Squared Error (RMSE).

    Args:
        y (np.ndarray): the real value of the points in the y axis coordinates
        y_hat (np.ndarray): the predicted value of the points in the y axis coordinates

    Returns:
        float: Root Mean Squared Error (RMSE)
    """
    return np.sqrt(np.mean(np.square(y - y_hat)))


def rmsle(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Computes the Root Mean Squared Log Error (RMSLE):
    $$
    RMSLE(y, \\hat{y}) = \\sqrt{\\frac{\\sum_{i=1}^{n}(\\log (y_i+1) - \\log (\\hat{y_i}+1))^2}{n}}
    $$

    Args:
        y (np.ndarray): the real value of the points in the y axis coordinates
        y_hat (np.ndarray): the predicted value of the points in the y axis coordinates

    Returns:
        float: Root Mean Squared Log Error (RMSLE)
    """
    return np.sqrt(np.mean(np.square((np.log(y+1) - np.log(y_hat+1)))))


def rmspe(y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-16) -> float:
    """
    Computes the Root Mean Squared Percentage Error (RMSPE).

    Args:
        y (np.ndarray): the real value of the points in the y axis coordinates
        y_hat (np.ndarray): the predicted value of the points in the y axis coordinates
        eps (float): eps value to prevent division by zero (default: 1E-16)

    Returns:
        float: Root Mean Squared Percentage Error (RMSPE)
    """
    return np.sqrt(np.mean(np.square((y - y_hat) / (y+eps))))


def rpd(y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-16) -> float:
    """
    Computes the Relative Percentage Difference (RPD).

    Args:
        y (np.ndarray): the real value of the points in the y axis coordinates
        y_hat (np.ndarray): the predicted value of the points in the y axis coordinates
        eps (float): eps value to prevent division by zero (default: 1E-16)

    Returns:
        float: Relative Percentage Difference (RPD)
    """
    return np.mean(np.abs((y - y_hat) / (np.maximum(y, y_hat)+eps)))


def residuals(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Computes the residual error of the fit.

    Args:
        y (np.ndarray): the real value of the points in the y axis coordinates
        y_hat (np.ndarray): the predicted value of the points in the y axis coordinates

    Returns:
        float: residual error of the fit
    """
    return np.sum(np.square((y-y_hat)))


def smape(y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-16) -> float:
    """
    Computes Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        y (np.ndarray): the real value of the points in the y axis coordinates
        y_hat (np.ndarray): the predicted value of the points in the y axis coordinates

    Returns:
        float: residual error of the fit
    """
    return np.mean(2.0 * np.abs(y_hat - y) / (np.abs(y) + np.abs(y_hat) + eps))