# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
import numpy as np


logger = logging.getLogger(__name__)


def linear_fit(x, y):
    m = (y[0] - y[-1])/(x[0] - x[-1])
    b = y[0] - (m*x[0])
    coef = (b, m)
    return coef


def linear_transform(x, coef):
    b, m = coef
    y_hat = x * m + b
    return y_hat


def linear_r2(x, y, coef):
    y_hat = linear_transform(x, coef)
    y_mean = np.mean(y)
    rss = np.sum((y-y_hat)**2)
    tss = np.sum((y-y_mean)**2)
    return 1.0 - (rss/tss)


def linear_residuals(x, y, coef):
    y_hat = linear_transform(x, coef)
    rss = np.sum((y-y_hat)**2)
    return rss