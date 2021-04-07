# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import logging
import numpy as np
from knee.rdp import get_r2, perpendicular_distance
from knee.linear_fit import linear_fit, linear_residuals, linear_r2
from uts import ema
import knee.multi_knee as mk
from enum import Enum


import cProfile


logger = logging.getLogger(__name__)


class Fit(Enum):
    best_fit = 'bestfit'
    point_fit = 'pointfit'

    def __str__(self):
        return self.value


def get_knee_binary(points):
    x = points[:,0]
    y = points[:,1]
    length = x[-1] - x[0]
    left = 0
    right = len(x) - 1

    # counter
    cnt = 0

    while  (right-left) > 1 and cnt < 10:
        logger.debug('Limits -> [%s, %s]', left, right)
        d = perpendicular_distance(points, left, right)
        middle = np.argmax(d)
        
        d = perpendicular_distance(points, left, middle)
        middle_left = np.argmax(d)

        d = perpendicular_distance(points, middle, right)
        middle_right = middle + np.argmax(d)

        logger.debug('IDX -> [%s, %s, %s]', middle_left, middle, middle_right)
        
        # middle point
        left_length = x[middle] - x[0]
        right_length = x[-1] - x[middle]
        coef_left = linear_fit(x[0:middle+1], y[0:middle+1])
        coef_right = linear_fit(x[middle:], y[middle:])
        r_left = linear_residuals(x[0:middle+1], y[0:middle+1], coef_left)
        r_rigth = linear_residuals(x[middle:], y[middle:], coef_right)
        error_middle = r_left*(left_length/length) + r_rigth*(right_length/length)

        # left middle point
        

        left_length = x[middle_left] - x[0]
        right_length = x[-1] - x[middle_left]

        coef_left = linear_fit(x[0:middle_left+1], y[0:middle_left+1])
        coef_right = linear_fit(x[middle_left:], y[middle_left:])
        r_left = linear_residuals(x[0:middle_left+1], y[0:middle_left+1], coef_left)
        r_rigth = linear_residuals(x[middle_left:], y[middle_left:], coef_right)
        error_middle_left = r_left*(left_length/length) + r_rigth*(right_length/length)


        # right middle point

        left_length = x[middle_right] - x[0]
        right_length = x[-1] - x[middle_right]

        coef_left = linear_fit(x[0:middle_right+1], y[0:middle_right+1])
        coef_right = linear_fit(x[middle_right:], y[middle_right:])
        r_left = linear_residuals(x[0:middle_right+1], y[0:middle_right+1], coef_left)
        r_rigth = linear_residuals(x[middle_right:], y[middle_right:], coef_right)
        error_middle_right = r_left*(left_length/length) + r_rigth*(right_length/length)

        logger.debug('Error -> (%s, %s, %s)', error_middle_left, error_middle, error_middle_right)

        if error_middle < error_middle_left:
            left = middle_left
            
        if error_middle < error_middle_right:
            right = middle_right
           
        cnt += 1
        logger.debug('New Limits -> [%s, %s]', left, right)


    logger.debug('C = %s', cnt)
    return (middle, 1, 1)


def get_knee(x, y, fit=Fit.point_fit):
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


def knee_points(points, fit=Fit.point_fit, debug=False):
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


# Version with cache
def knee_cached(x, y, cache_left: dict, cache_right: dict, shift, debug=False):
    index = 2
    length = x[-1] - x[0]
    left_length = x[index] - x[0]
    right_length = x[-1] - x[index]

    key = index + shift

    if key in cache_left:
        logger.debug('Cache Left Hit (%s, %s)', key, index)
        r_left = cache_left[key]
    else:
        logger.debug('Cache Left Miss (%s, %s)', key, index)
        coef_left = linear_fit(x[0:index+1], y[0:index+1])
        r_left = linear_residuals(x[0:index+1], y[0:index+1], coef_left)
        cache_left[key] = r_left
        
    if key in cache_right:
        logger.debug('Cache Right Hit (%s, %s)', key, index)
        r_rigth = cache_right[key]
    else:
        logger.debug('Cache Right Miss (%s, %s)', key, index)
        coef_right = linear_fit(x[index:], y[index:])
        r_rigth = linear_residuals(x[index:], y[index:], coef_right)
        cache_right[key] = r_rigth
    
    error = r_left*(left_length/length) + r_rigth*(right_length/length)
        

    for i in range(index+1, len(x)-2):
        left_length = x[i] - x[0]
        right_length = x[-1] - x[i]
        
        key = i + shift

        if key in cache_left:
            logger.debug('Cache Left Hit (%s, %s)', key, i)
            r_left = cache_left[key]
        else:
            logger.debug('Cache Left Miss (%s, %s)', key, i)
            coef_left = linear_fit(x[0:i+1], y[0:i+1])
            r_left = linear_residuals(x[0:i+1], y[0:i+1], coef_left)
            cache_left[key] = r_left
        
        if key in cache_right:
            logger.debug('Cache Right Hit (%s, %s)', key, i)
            r_rigth = cache_right[key]
        else:
            logger.debug('Cache Right Miss (%s, %s)', key, i)
            coef_right = linear_fit(x[i:], y[i:])
            r_rigth = linear_residuals(x[i:], y[i:], coef_right)
            cache_right[key] = r_rigth
            
            
        current_error = r_left*(left_length/length) + r_rigth*(right_length/length)
            
        if current_error < error:
            error = current_error
            index = i
            #coef_left = i_coef_left
            #coef_right = i_coef_right
        
    #return index
    if debug:
        return {'knee': index, 'coef_left': 0, 'coef_right': 0}
    else:
        return index


def multiknee_rec_cached(x, y, left, right, t, cache_left: dict, cache_right: dict, shift: int, debug: bool):
    logger.debug('Cache Left: %s', cache_left)
    logger.debug('Cache Right: %s', cache_right)
    response = knee_cached(x[left:right], y[left:right], cache_left, cache_right, shift, debug)
    
    if debug is False:
        response_idx = response
        knee_idx = response + left
        response = knee_idx
    else:
        response_idx = response['knee']
        knee_idx = response['knee'] + left
        response['knee'] = knee_idx
        response['left'] = left
        response['right'] = right
    
    # Split the cache into left and right
    values_left = [(k, v) for k, v in cache_left.items() if k < response_idx]
    values_right = [(k, v) for k, v in cache_right.items() if k > response_idx]
    #values.sort(key=lambda tup: tup[0])
    #left_values = [(k, v) for k,v in values if k < response_idx]
    #right_values= [(k, v) for k,v in values if k > response_idx]

    cache_left = dict(values_left)
    cache_right = dict(values_right)

    logger.info('Multi Knee Rec [%s, %s] = %s', left, right, knee_idx)
    
    coef_left = linear_fit(x[left:knee_idx+1], y[left:knee_idx+1])
    coef_right = linear_fit(x[knee_idx+1:right], y[knee_idx+1:right])
    r2_left = linear_r2(x[left:knee_idx+1], y[left:knee_idx+1], coef_left)
    r2_right = linear_r2(x[knee_idx+1:right], y[knee_idx+1:right], coef_right)

    #logger.info('R2[%s, %s] Left Part [%s, %s] Right Part [%s, %s]', r2_left, r2_right, left, knee_idx+1, knee_idx+1, right)
    #input('wait...')

    if r2_left <= t and (knee_idx+1-left) > 4:
        logger.debug('MK (Left) [%s, %s]', left, knee_idx+1)
        responce_left = multiknee_rec_cached(x, y, left, knee_idx+1, t, cache_left, {}, 0, debug)
        rv = responce_left
        rv.append(response)
    else:
        rv = [response]

    if r2_right <= t and (right-(knee_idx+1)) > 4:
        logger.debug('MK (Right) [%s, %s]', knee_idx+1, right)
        response_right = multiknee_rec_cached(x, y, knee_idx+1, right, t, {}, cache_right, response_idx, debug)
        rv.extend(response_right)

    return rv


def multiknee_cache(points, t = 0.9, debug=False):
    Ds = ema.linear(points, 1.0)
    pmin = Ds.min(axis = 0)
    pmax = Ds.max(axis = 0)
    Dn = (Ds - pmin)/(pmax - pmin)

    x = Dn[:,0]
    y = Dn[:,1]

    coef = linear_fit(x, y)
    if linear_r2(x, y, coef) <= t:
        cache_left = {}
        cache_right = {}
        response = multiknee_rec_cached(x, y, 0, len(points), t, cache_left, cache_right, 0, debug)
        return {'knees': response, 'Ds':Ds, 'Dn': Dn}
    else:
        return {'knees': [], 'Ds':Ds, 'Dn': Dn}


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


def multiknee(points, t = 0.99, fit=Fit.point_fit, debug=False):
    Ds = ema.linear(points, 1.0)
    pmin = Ds.min(axis = 0)
    pmax = Ds.max(axis = 0)
    Dn = (Ds - pmin)/(pmax - pmin)

    x = Dn[:,0]
    y = Dn[:,1]

    coef = linear_fit(x, y)
    if linear_r2(x, y, coef) <= t:
        response = multiknee_rec(x, y, 0, len(points), t, fit, debug)
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
    return mk.multi_knee(knee_points, points, t1, t2)