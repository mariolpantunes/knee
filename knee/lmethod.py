# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import logging
import numpy as np
from rdp import get_r2
from linear_fit import linear_fit, linear_residuals, linear_r2
from uts import ema
from enum import Enum


logger = logging.getLogger(__name__)


class Fit(Enum):
    best_fit = 'bestfit'
    point_fit = 'pointfit'

    def __str__(self):
        return self.value


def get_knee(x, y, fit):
    index = 2
    length = x[-1] - x[0]    
    left_length = x[index] - x[0]
    right_length = x[-1] - x[index]

    if fit is Fit.best_fit:
        coef_left, r_left, *other  = np.polyfit(x[0:index+1], y[0:index+1], 1, full=True)
        coef_right, r_rigth, *other = np.polyfit(x[index:], y[index:], 1, full=True)
        error = r_left[0]*(left_length/length) + r_rigth[0]*(right_length/length)
        #error = (r_left[0] + r_rigth[0]) / 2.0
    else:
        coef_left = linear_fit(x[0:index+1], y[0:index+1])
        coef_right = linear_fit(x[index:], y[index:])
        r_left = linear_residuals(x[0:index+1], y[0:index+1], coef_left)
        r_rigth = linear_residuals(x[index:], y[index:], coef_right)
        error = r_left*(left_length/length) + r_rigth*(right_length/length)
        #error = (r_left + r_rigth) / 2.0
    
    #logger.info("Error(%s) = %s", index, error)

    for i in range(index+1, len(x)-2):
        left_length = x[i] - x[0]
        right_length = x[-1] - x[i]

        if fit is Fit.best_fit:
            i_coef_left, r_left, *other  = np.polyfit(x[0:i+1], y[0:i+1], 1, full=True)
            i_coef_right, r_rigth, *other = np.polyfit(x[i:], y[i:], 1, full=True)
            current_error = r_left[0]*(left_length/length) + r_rigth[0]*(right_length/length)
            #current_error = (r_left[0] + r_rigth[0]) / 2.0
        else:
            i_coef_left = linear_fit(x[0:i+1], y[0:i+1])
            i_coef_right = linear_fit(x[i:], y[i:])
            r_left = linear_residuals(x[0:i+1], y[0:i+1], i_coef_left)
            r_rigth = linear_residuals(x[i:], y[i:], i_coef_right)
            current_error = r_left*(left_length/length) + r_rigth*(right_length/length)
            #current_error = (r_left + r_rigth) / 2.0
        
        #logger.info("Error(%s) = %s", i, error)

        if current_error < error:
            error = current_error
            index = i
            coef_left = i_coef_left
            coef_right = i_coef_right

    return (index, coef_left, coef_right)


def knee_points(points, fit=Fit.best_fit, debug=False):
    x = points[:,0]
    y = points[:,1]
    return knee(x, y, fit, debug)


def knee(x, y, fit, debug=False):

    last_knee = -1
    cutoff  = current_knee = len(x)
    #logger.info('Knee [%s, %s, %s]', cutoff, last_knee, current_knee)

    done = False
    while current_knee != last_knee and not done:
        last_knee = current_knee
        current_knee, coef_left, coef_right = get_knee(x[0:cutoff+1], y[0:cutoff+1], fit)
        #input('wait...')
        cutoff = int((current_knee + last_knee)/2)
        if cutoff < 10:
            done = True
        
    #return index
    if debug:
        return {'knee': current_knee, 'coef_left': coef_left, 'coef_right': coef_right}
    else:
        return current_knee


def multiknee_rec(x, y, left, right, t, fit, debug):
    response = knee(x[left:right], y[left:right], fit, debug)
    if debug is False:
        knee_idx = response + left
        response = knee_idx
    else:
        knee_idx = response['knee'] + left
        response['knee'] = knee_idx
        response['left'] = left
        response['right'] = right
    
    logger.info('Multi Knee Rec [%s, %s] = %s', left, right, knee_idx)
    
    coef_left = linear_fit(x[left:knee_idx+1], y[left:knee_idx+1])
    coef_right = linear_fit(x[knee_idx+1:right], y[knee_idx+1:right])
    r2_left = linear_r2(x[left:knee_idx+1], y[left:knee_idx+1], coef_left)
    r2_right = linear_r2(x[knee_idx+1:right], y[knee_idx+1:right], coef_right)

    logger.info('R2[%s, %s] Left Part [%s, %s] Right Part [%s, %s]', r2_left, r2_right, left, knee_idx+1, knee_idx+1, right)

    #input('wait...')

    if r2_left <= t and (knee_idx+1-left) > 4:
        responce_left = multiknee_rec(x, y, left, knee_idx+1, t, fit, debug)
        rv = responce_left
        rv.append(response)
    else:
        rv = [response]

    if r2_right <= t and (right-(knee_idx+1)) > 4:
        response_right = multiknee_rec(x, y, knee_idx+1, right, t, fit, debug)
        rv.extend(response_right)

    return rv


def multiknee(points, t = 0.9, fit=Fit.point_fit, debug=False):
    Ds = ema.linear(points, 1.0)
    pmin = Ds.min(axis = 0)
    pmax = Ds.max(axis = 0)
    Dn = (Ds - pmin)/(pmax - pmin)

    x = Dn[:,0]
    y = Dn[:,1]

    coef = linear_fit(x, y)
    if linear_r2(x, y, coef) <= t:
        response = multiknee_rec(x, y, 0, len(points), t, fit, debug)
        return {'knees': response, 'Ds':Ds, 'Dn': Dn}
    else:
        return {'knees': [], 'Ds':Ds, 'Dn': Dn}

#l = [[1.0, 1.0],[2, 0.25],[3, 0.111],[4, 0.0625],[5, 0.04],[6, 0.0277777],[7, 0.0204],[8, 0.015625],[9, 0.012345679],[10, .01]]
#points = np.array(l)
#print(points)

#knees = knee(points)
#print("Knee:", knees)