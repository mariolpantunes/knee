# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import logging
import numpy as np


def performance(points: np.ndarray, knees: np.ndarray):
    x = points[:,0]
    y = points[:,1]

    distances_x = []
    distances_y = []

    total_x = math.fabs(x[-1] - x[0])
    total_y = math.fabs(y[-1] - y[0])

    previous_knee_x =x[knees[0]]
    previous_knee_y =y[knees[0]]

    distances_x.append(math.fabs(x[0]-previous_knee_x))
    distances_y.append(math.fabs(y[0]-previous_knee_y))

    for i in range(1, len(knees)):
        knee_x = x[knees[i]]
        knee_y = y[knees[i]]

        distances_x.append(math.fabs(previous_knee_x - knee_x))
        distances_y.append(math.fabs(previous_knee_y - knee_y))

        previous_knee_x = knee_x
        previous_knee_y = knee_y
    
    distances_x = np.array(distances_x)
    distances_y = np.array(distances_y)

    average_x = np.median(distances_x)/total_x
    average_y = np.median(distances_y)/total_y

    return (average_x, average_y, average_y/average_x)

