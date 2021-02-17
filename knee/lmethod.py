# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import math
import numpy as np
from uts import ema

def get_knee(x,y):
    index = 2
    coef_left, r_left, *other  = np.polyfit(x[0:index+1], y[0:index+1], 1, full=True)
    coef_right, r_rigth, *other = np.polyfit(x[index:], y[index:], 1, full=True)
    error = (r_left[0] + r_rigth[0]) / 2.0

    for i in range(index+1, len(x)-2):
        i_coef_left, r_left, *other  = np.polyfit(x[0:i+1], y[0:i+1], 1, full=True)
        i_coef_right, r_rigth, *other = np.polyfit(x[i:], y[i:], 1, full=True)
        current_error = (r_left[0] + r_rigth[0]) / 2.0

        if current_error < error:
            error = current_error
            index = i
            coef_left = i_coef_left
            coef_right = i_coef_right

    return (index, coef_left, coef_right)

def knee(points, debug=False):
    Ds = ema.linear(points, 1.0)
    pmin = Ds.min(axis = 0)
    pmax = Ds.max(axis = 0)
    Dn = (Ds - pmin)/(pmax - pmin)

    x = Dn[:,0]
    y = Dn[:,1]

    last_knee = -1
    cutoff  = current_knee = len(points)
    print('({}, {}, {})'.format(cutoff, last_knee, current_knee))

    while current_knee != last_knee:
        last_knee = current_knee
        current_knee, coef_left, coef_right = get_knee(x[0:cutoff], y[0:cutoff])
        cutoff = int((current_knee + last_knee)/2)
        print('({}, {}, {})'.format(cutoff, last_knee, current_knee))
    
    #return index
    if debug:
        return {'knees': np.array([current_knee]), 'left': coef_left, 'right': coef_right, 'Dn': Dn}
    else:
        return np.array([current_knee])


def multiknee(points, debug=False):
    Ds = ema.linear(points, 1.0)
    pmin = Ds.min(axis = 0)
    pmax = Ds.max(axis = 0)
    Dn = (Ds - pmin)/(pmax - pmin)

    x = Dn[:,0]
    y = Dn[:,1]


#l = [[1.0, 1.0],[2, 0.25],[3, 0.111],[4, 0.0625],[5, 0.04],[6, 0.0277777],[7, 0.0204],[8, 0.015625],[9, 0.012345679],[10, .01]]
#points = np.array(l)
#print(points)

#knees = knee(points)
#print("Knee:", knees)