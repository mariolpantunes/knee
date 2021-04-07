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


def get_knee(x, y):
    gradient = np.gradient(y, x, edge_order=1)
    t = thresholding.isodata(gradient)
    diff = np.absolute(gradient - t)
    knee = np.argmin(diff[1:-1]) + 1

    return knee


def knee_points(points):
    x = points[:,0]
    y = points[:,1]

    gradient = np.gradient(y, x, edge_order=1)
    t = thresholding.isodata(gradient)
    diff = np.absolute(gradient - t)
    
    knee = cutoff = 0 #np.argmin(diff)
    last_knee = -1

    while last_knee < knee and (len(x)-cutoff) > 2:
        last_knee = knee
        knee = argmin(diff[cutoff:-1]) + cutoff
        cutoff = int(math.ceil(knee/2.0))
        
    return knee

def multiknee_rec(x, y, left, right, t, debug):
    logger.info('MultiKnee [%s, %s]', left, right)
    response = get_knee(x[left:right], y[left:right], debug)
    knee_idx = response + left
    response = knee_idx

    if (knee_idx+1-left) > 3:
        
        coef_left = linear_fit(x[left:knee_idx+1], y[left:knee_idx+1])
        r2_left = linear_r2(x[left:knee_idx+1], y[left:knee_idx+1], coef_left)
        if r2_left <= t:
            logger.info('Left')
            responce_left = multiknee_rec(x, y, left, knee_idx, t, debug)
            rv = responce_left
            rv.append(response)
        else:
            rv = [response]
    else:
        rv = [response]

    #logger.info('R2[%s, %s] Left Part [%s, %s] Right Part [%s, %s]', r2_left, r2_right, left, knee_idx+1, knee_idx+1, right)

    #if r2_left < 0:
    #    logger.error('Coef %s Points [%s, %s]', coef_left, (x[left], y[left]),(x[knee_idx], y[knee_idx]))
    #    logger.error('%s', x[left:knee_idx+1])
    #    logger.error('%s', y[left:knee_idx+1])


    if (right-(knee_idx+1)) > 3:
        coef_right = linear_fit(x[knee_idx+1:right], y[knee_idx+1:right])
        r2_right = linear_r2(x[knee_idx+1:right], y[knee_idx+1:right], coef_right)

        if r2_right <= t:
            logger.info('Right')
            response_right = multiknee_rec(x, y, knee_idx+1, right, t, debug)
            rv.extend(response_right)
    return rv


def multiknee(points, t = 0.99, debug=False):
    Ds = ema.linear(points, 1.0)
    pmin = Ds.min(axis = 0)
    pmax = Ds.max(axis = 0)
    Dn = (Ds - pmin)/(pmax - pmin)

    x = Dn[:,0]
    y = Dn[:,1]

    coef = linear_fit(x, y)
    if linear_r2(x, y, coef) <= t:
        response = multiknee_rec(x, y, 0, len(points), t, debug)
        if debug:
            return {'knees': response, 'Ds':Ds, 'Dn': Dn}
        else:
            return np.array(response)
    else:
        if debug:
            return {'knees': np.array([]), 'Ds':Ds, 'Dn': Dn}
        else:
            np.array([])


def multi_knee(points, t1=0.99, t2=2):
    return mk.multi_knee(knee_points, points, t1, t2)