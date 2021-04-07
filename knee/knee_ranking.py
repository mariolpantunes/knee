# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import uts
import math
import logging
import numpy as np
import knee.rdp as rdp
import knee.menger


logger = logging.getLogger(__name__)


def distance_to_similarity(array: np.ndarray) -> np.ndarray:
    return max(array) - array


def rank(array: np.ndarray) -> np.ndarray:
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks


def slope_ranking(points: np.ndarray, knees: np.ndarray, t=0.8, relative=True) -> np.ndarray:
    rankings = []

    x = points[:,0]
    y = points[:,1]
    
    #print('Slope {}'.format(0))
    j = rdp.straight_line(points, 0, knees[0], t)
    
    slope = (y[j]-y[knees[0]]) / (x[j]-x[knees[0]])
    rankings.append(math.fabs(slope))
    
    for i in range(1, len(knees)):
        #print('Slope {}'.format(i))
        j = rdp.straight_line(points, knees[i-1], knees[i], t)
        
        slope = (y[j]-y[knees[i]]) / (x[j]-x[knees[i]])
        rankings.append(math.fabs(slope))

    rankings = np.array(rankings)

    if relative:
        rankings = rank(rankings)
        # Min Max normalization
        rankings = (rankings - np.min(rankings))/np.ptp(rankings)
    else:
        # Standardization (Z-score Normalization)
        rankings = (rankings - np.mean(rankings))/np.std(rankings)

    return rankings


def weighted_slope_ranking(points: np.ndarray, knees: np.ndarray, t=0.8, relative=True) -> np.ndarray:
    rankings = []
    weights  = []

    x = points[:,0]
    y = points[:,1]
    
    #print('Slope {}'.format(0))
    j = rdp.straight_line(points, 0, knees[0], t)
    slope = (y[j]-y[knees[0]]) / (x[j]-x[knees[0]])
    rankings.append(math.fabs(slope))
    # height of the segment
    d = math.fabs(y[j] - y[knees[0]])
    weights.append(d)
    
    for i in range(1, len(knees)):
        #print('Slope {}'.format(i))
        j = rdp.straight_line(points, knees[i-1], knees[i], t)
        
        slope = (y[j]-y[knees[i]]) / (x[j]-x[knees[i]])
        rankings.append(math.fabs(slope))
        # height of the segment
        d = math.fabs(y[j] - y[knees[0]])
        weights.append(d)

    rankings = np.array(rankings)
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    rankings = rankings * weights

    if relative:
        rankings = rank(rankings)
        # Min Max normalization
        rankings = (rankings - np.min(rankings))/np.ptp(rankings)
    else:
        # Standardization (Z-score Normalization)
        rankings = (rankings - np.mean(rankings))/np.std(rankings)

    return rankings