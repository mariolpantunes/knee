# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import numpy as np


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))


def zscore(xi, mean, std):
    return (xi-mean)/std


def zscore_points(xi, points):
    xpoints = np.transpose(points)[0]
    ypoints = np.transpose(points)[1]
    
    values = ypoints
    tmp = xpoints[1:] - xpoints[:-1]
    weights = np.concatenate(([tmp.min()], tmp))
    mean, std = weighted_avg_and_std(values, weights)
    return zscore(xi, mean, std)


#l = [[1.0, 1.0],[2, 0.25],[3, 0.111],[4, 0.0625],[5, 0.04],[6, 0.0277777],[7, 0.0204],[8, 0.015625],[9, 0.012345679],[10, .01]]
#points = np.array(l)
#print(points)
#print(zscore_points(1, points))