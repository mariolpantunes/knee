# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import math
import numpy as np
from uts import ema


def knee(points, debug=False):
    Ds = ema.linear(points, 1.0)
    pmin = Ds.min(axis = 0)
    pmax = Ds.max(axis = 0)
    Dn = (Ds - pmin)/(pmax - pmin)

    DnT = np.transpose(Dn)

    x = DnT[0]
    y = DnT[1]

    index = 2
    coef_left, r_left, *other  = np.polyfit(x[0:index+1], y[0:index+1], 1, full=True)
    coef_right, r_rigth, *other = np.polyfit(x[index:], y[index:], 1, full=True)
    error = (r_left[0] + r_rigth[0]) / 2.0

    for i in range(index+1, len(Dn)-2):
        i_coef_left, r_left, *other  = np.polyfit(x[0:i+1], y[0:i+1], 1, full=True)
        i_coef_right, r_rigth, *other = np.polyfit(x[i:], y[i:], 1, full=True)
        current_error = (r_left[0] + r_rigth[0]) / 2.0

        if current_error < error:
            error = current_error
            index = i
            coef_left = i_coef_left
            coef_right = i_coef_right
    
    knees = [index]
    
    #return index
    if debug:
        return {'knees': np.array(knees), 'left': coef_left, 'right': coef_right, 'Ds':Ds, 'Dn': Dn, 'Dd':Dn}
    else:
        return np.array(knees)

#l = [[1.0, 1.0],[2, 0.25],[3, 0.111],[4, 0.0625],[5, 0.04],[6, 0.0277777],[7, 0.0204],[8, 0.015625],[9, 0.012345679],[10, .01]]
#points = np.array(l)
#print(points)

#knees = knee(points)
#print("Knee:", knees)