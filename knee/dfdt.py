# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import logging
import numpy as np
from numpy.core.fromnumeric import argmin
from knee.linear_fit import linear_fit, linear_r2
from uts import thresholding, ema
import knee.multi_knee as mk


logger = logging.getLogger(__name__)


def get_knee(x: np.ndarray, y: np.ndarray) -> int:
    """Return the index of the knee point based on the DFDT method.

    Keyword arguments:
    x -- x axis coordinates
    y -- y axis coordinates
    """
    gradient = np.gradient(y, x, edge_order=1)
    return get_knee_gradient(gradient)


def get_knee_gradient(gradient: np.ndarray) -> int:
    """Return the index of the knee point based on the DFDT method.

    Keyword arguments:
    gradient -- the first order gradient of the points
    """
    t = thresholding.isodata(gradient)
    diff = np.absolute(gradient - t)
    knee = np.argmin(diff[1:-1]) + 1
    return knee


def knee_points(points: np.ndarray) -> int:
    """Returns the index of the knee point based on the DFDT method.
    It uses the iterative refinement  method.

    Keyword arguments:
    points -- numpy array with the points (x, y)
    """
    x = points[:,0]
    y = points[:,1]

    gradient = np.gradient(y, x, edge_order=1)
    
    knee = cutoff = 0 
    last_knee = -1

    while last_knee < knee and (len(x)-cutoff) > 2:
        last_knee = knee
        knee = get_knee_gradient(gradient[cutoff:]) + cutoff
        cutoff = int(math.ceil(knee/2.0))
        
    return knee


def multi_knee(points, t1=0.99, t2=2):
    """Recursive knee point detection based on DFDT.
    It returns the knee points on the curve.
    
    Keyword arguments:
    points -- numpy array with the points (x, y)
    t1 -- coefficient of determination threshold (default 0.99)
    t2 -- number of points threshold (default 2)
    """
    return mk.multi_knee(knee_points, points, t1, t2)