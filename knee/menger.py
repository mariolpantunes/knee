# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import logging
import numpy as np
from knee.linear_fit import linear_fit, linear_r2
from uts import ema
import knee.multi_knee as mk


logger = logging.getLogger(__name__)


def menger_curvature(f, g, h):
    x1 = f[0]
    y1 = f[1]
    x2 = g[0]
    y2 = g[1]
    x3 = h[0]
    y3 = h[1]

    nom = 2.0 * math.fabs((x2-x1)*(y3-y2))-((y2-y1)*(x3-x2))
    temp = math.fabs((x2-x1)**2.0 + (y2-y1)**2.0)*math.fabs((x3-x2)**2.0 + (y3-y2)**2.0) * math.fabs((x1-x3)**2.0 + (y1-y3)**2.0)
    dem = math.sqrt(temp)

    return nom/dem


def get_knee(points, debug=False):
    curvature = [0]

    for i in range(1, len(points)-1):
        f = points[i]
        g = points[i-1]
        h = points[i+1]

        curvature.append(menger_curvature(f, g, h))
    
    curvature.append(0)
    curvature = np.array(curvature)
    return np.argmax(curvature)


def multiknee_rec(points, left, right, t, debug):
    logger.info('MultiKnee [%s, %s]', left, right)
    response = get_knee(points[left:right], debug)
    knee_idx = response + left
    response = knee_idx

    x = points[:,0]
    y = points[:,1]

    if (knee_idx+1-left) > 4:
        
        coef_left = linear_fit(x[left:knee_idx+1], y[left:knee_idx+1])
        r2_left = linear_r2(x[left:knee_idx+1], y[left:knee_idx+1], coef_left)
        if r2_left <= t:
            responce_left = multiknee_rec(points, left, knee_idx, t, debug)
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


    if (right-(knee_idx+1)) > 4:
        coef_right = linear_fit(x[knee_idx+1:right], y[knee_idx+1:right])
        r2_right = linear_r2(x[knee_idx+1:right], y[knee_idx+1:right], coef_right)

        if r2_right <= t:
            response_right = multiknee_rec(points, knee_idx+1, right, t, debug)
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
        response = multiknee_rec(Dn, 0, len(points), t, debug)
        if debug:
            return {'knees': response, 'Ds':Ds, 'Dn': Dn}
        else:
            return np.array(response)
    else:
        if debug:
            return {'knees': np.array([]), 'Ds':Ds, 'Dn': Dn}
        else:
            np.array([])


def multi_knee(points, t1=0.99, t2=4):
    return mk.multi_knee(get_knee, points, t1, t2)