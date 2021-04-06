# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
import numpy as np
from knee.linear_fit import linear_fit_points, linear_r2_points


logger = logging.getLogger(__name__)


def multi_knee(get_knee, points, t1=0.99, t2=3):
    return np.array(multi_knee_rec(get_knee, points, 0, len(points), t1, t2))


def multi_knee_rec(get_knee, points, left, right, t1, t2):
    #x = points[left:right,0]
    #y = points[left:right,1]

    pt = points[left:right]

    coef = linear_fit_points(pt)
    if linear_r2_points(pt, coef) < t1 and len(points) >= t2:
        idx = get_knee(pt) + left
        left_knees = multi_knee_rec(get_knee, points, left, idx+1, t1, t2)
        right_knees = multi_knee_rec(get_knee, points, idx+1, right, t1, t2)

        return left_knees + [idx] + right_knees
    else:
        return []


