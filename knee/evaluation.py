# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import logging
import numpy as np
import knee.linear_fit as lf


def get_neighbourhood_points(points: np.ndarray, a: int, b: int, t: float = 0.95):
    x = points[:,0]
    y = points[:,1]
    return get_neighbourhood(x, y, a, b, t)


def get_neighbourhood(x: np.ndarray, y: np.ndarray, a: int, b: int, t: float = 0.95):
    r2 = 1.0
    i = a - 1
    slope = 1.0
    
    while r2 > t and i > b:
        previous_res = (i, r2, slope)
        i -= 1
        coef = lf.linear_fit(x[i:a+1], y[i:a+1])
        r2 = lf.linear_r2(x[i:a+1], y[i:a+1], coef)
        _, slope = coef

    if r2 > t:
        return i, r2, slope
    else:
        return previous_res


def performance_individual(points: np.ndarray, knees: np.ndarray) -> tuple[float, float, float]:
    x = points[:,0]
    y = points[:,1]

    slopes = []
    coeffients = []

    previous_knee = 0
    for i in range(0, len(knees)):
        _, r2, slope  = get_neighbourhood(x, y, knees[i], previous_knee)
        coeffients.append(r2)
        slopes.append(math.fabs(slope))
        previous_knee = knees[i]
    
    slopes = np.array(slopes)
    slopes = slopes/slopes.max()

    coeffients = np.array(coeffients)
    coeffients = coeffients/coeffients.max()

    average_slope = np.average(slopes)
    average_coeffients = np.average(coeffients)

    cost = 1.0 / (average_slope * average_coeffients) 

    return average_slope, average_coeffients, cost


def performance(points: np.ndarray, knees: np.ndarray) -> tuple[float, float, float, float, float]:
    x = points[:,0]
    y = points[:,1]

    distances_x = []
    distances_y = []
    slopes = []
    coeffients = []

    total_x = math.fabs(x[-1] - x[0])
    total_y = math.fabs(y[-1] - y[0])

    previous_knee_x = x[knees[0]]
    previous_knee_y = y[knees[0]]

    delta_x = x[0] - previous_knee_x
    delta_y = y[0] - previous_knee_y
    distances_x.append(math.fabs(delta_x))
    distances_y.append(math.fabs(delta_y))
    slopes.append(math.fabs(delta_y/delta_x))

    coef = lf.linear_fit(x[0:knees[0]+1], y[0:knees[0]+1])
    r2 = lf.linear_r2(x[0:knees[0]+1], y[0:knees[0]+1], coef)
    coeffients.append(r2)

    for i in range(1, len(knees)):
        knee_x = x[knees[i]]
        knee_y = y[knees[i]]

        delta_x = previous_knee_x - knee_x
        delta_y = previous_knee_y - knee_y

        coef = lf.linear_fit(x[knees[i-1]:knees[i]+1], y[knees[i-1]:knees[i]+1])
        r2 = lf.linear_r2(x[knees[i-1]:knees[i]+1], y[knees[i-1]:knees[i]+1], coef)
    
        distances_x.append(math.fabs(delta_x))
        distances_y.append(math.fabs(delta_y))
        slopes.append(math.fabs(delta_y/delta_x))
        coeffients.append(r2)

        previous_knee_x = knee_x
        previous_knee_y = knee_y
    
    distances_x = np.array(distances_x)
    distances_y = np.array(distances_y)
    slopes = np.array(slopes)
    slopes = slopes/slopes.max()

    coeffients = np.array(coeffients)
    coeffients = coeffients/coeffients.max()

    p = slopes * distances_y * coeffients

    average_x = np.average(distances_x)/total_x
    average_y = np.average(distances_y)/total_y
    average_slope = np.average(slopes)
    average_coeffients = np.average(coeffients)

    p = np.average(p) 

    return (average_x, average_y, average_slope, average_coeffients, p)

