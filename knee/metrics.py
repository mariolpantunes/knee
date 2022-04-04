# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import numpy as np


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