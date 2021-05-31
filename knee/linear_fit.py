# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
import math
import numpy as np


logger = logging.getLogger(__name__)


def linear_fit_points(points):
    x = points[:, 0]
    y = points[:, 1]
    return linear_fit(x, y)


def linear_fit(x, y):
    m = (y[0] - y[-1])/(x[0] - x[-1])
    b = y[0] - (m*x[0])
    coef = (b, m)
    return coef


def linear_transform(x, coef):
    b, m = coef
    y_hat = x * m + b
    return y_hat


def linear_r2_points(points, coef):
    x = points[:, 0]
    y = points[:, 1]
    return linear_r2(x, y, coef)


def linear_r2(x, y, coef):
    y_hat = linear_transform(x, coef)
    y_mean = np.mean(y)
    rss = np.sum((y-y_hat)**2)
    tss = np.sum((y-y_mean)**2)
    if tss == 0:
        #print(f'Error ({coef}) -> {x} {y} {y} {y_hat}')
        return  1.0 - rss
    return 1.0 - (rss/tss)


def linear_residuals(x, y, coef):
    y_hat = linear_transform(x, coef)
    rss = np.sum((y-y_hat)**2)
    return rss