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
    slopes = []

    total_x = math.fabs(x[-1] - x[0])
    total_y = math.fabs(y[-1] - y[0])

    previous_knee_x =x[knees[0]]
    previous_knee_y =y[knees[0]]

    delta_x = x[0] - previous_knee_x
    delta_y = y[0] - previous_knee_y
    distances_x.append(math.fabs(delta_x))
    distances_y.append(math.fabs(delta_y))
    slopes.append(math.fabs(delta_y/delta_x))

    for i in range(1, len(knees)):
        knee_x = x[knees[i]]
        knee_y = y[knees[i]]

        delta_x = previous_knee_x - knee_x
        delta_y = previous_knee_y - knee_y

        distances_x.append(math.fabs(delta_x))
        distances_y.append(math.fabs(delta_y))
        slopes.append(math.fabs(delta_y/delta_x))

        previous_knee_x = knee_x
        previous_knee_y = knee_y
    
    distances_x = np.array(distances_x)
    distances_y = np.array(distances_y)
    slopes = np.array(slopes)
    slopes = slopes/slopes.max()

    average_x = np.average(distances_x)/total_x
    average_y = np.average(distances_y)/total_y
    average_slope = np.average(slopes)

    return (average_x, average_y, average_slope)

