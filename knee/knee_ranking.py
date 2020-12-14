# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import numpy as np
import math


def rank(array):
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks


def curvature_ranking(points: np.ndarray, knees: np.ndarray) -> np.ndarray:
    pt = np.transpose(points)

    x = pt[0]
    y = pt[1]

    gradient1 = np.gradient(y, x, edge_order=1)
    gradient2 = np.gradient(y, x, edge_order=2)

    rankings = []
    for idx in knees:
        curvature = math.fabs(gradient2[idx]) / (1.0 + gradient1[idx]**2.0)**(1.5)
        rankings.append(curvature)
    
    rankings = rank(np.array(rankings))
    rankings = (rankings - np.min(rankings))/np.ptp(rankings)

    return rankings


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


def menger_curvature_ranking(points: np.ndarray, knees: np.ndarray) -> np.ndarray:

    rankings = []
    for idx in knees:
        f = points[idx]
        g = points[idx-1]
        h = points[idx+1]
        curvature = menger_curvature(f, g, h)
        rankings.append(curvature)
    
    rankings = rank(np.array(rankings))
    rankings = (rankings - np.min(rankings))/np.ptp(rankings)

    return rankings