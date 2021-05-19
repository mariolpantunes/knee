# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


#import linear_fit, linear_r2

import math
import enum
import logging
import numpy as np
import knee.rdp as rdp 
import knee.linear_fit


class ClusterRanking(enum.Enum):
    left = 'left'
    linear = 'linear'
    right = 'right'

    def __str__(self):
        return self.value


logger = logging.getLogger(__name__)


def distance_to_similarity(array: np.ndarray) -> np.ndarray:
    return max(array) - array


def rank(array: np.ndarray) -> np.ndarray:
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))

    return ranks


def slope_ranking(points: np.ndarray, knees: np.ndarray, t=0.8, relative=True) -> np.ndarray:
    if len(knees) == 1.0:
        rankings = np.array([1.0])
    else:
        rankings = []

        x = points[:,0]
        y = points[:,1]
        
        #print('Slope {}'.format(0))
        j = rdp.straight_line(points, 0, knees[0], t)
       
        if j == knees[0]:
            rankings.append(0.0)
        else:
            slope = (y[j]-y[knees[0]]) / (x[j]-x[knees[0]])
            rankings.append(math.fabs(slope))
        
        for i in range(1, len(knees)):
            #print('Slope {}'.format(i))
            j = rdp.straight_line(points, knees[i-1], knees[i], t)
            
            if j == knees[i]:
                 rankings.append(0.0)
            else:
                slope = (y[j]-y[knees[i]]) / (x[j]-x[knees[i]])
                rankings.append(math.fabs(slope))

        rankings = np.array(rankings)

        if relative:
            rankings = rank(rankings)
            # Min Max normalization
            if len(rankings) > 1:
                rankings = (rankings - np.min(rankings))/np.ptp(rankings)
            else:
                rankings = np.array([1.0])
        else:
            # Standardization (Z-score Normalization)
            rankings = (rankings - np.mean(rankings))/np.std(rankings)

    return rankings


'''def cluster_ranking(points: np.ndarray, knees: np.ndarray, relative=True) -> np.ndarray:
    np.set_printoptions(precision=3)
    rankings = []
    weights  = []
    #distances = []

    x = points[:,0]
    y = points[:,1]

    #print('Slope {}'.format(0))
    #j = rdp.naive_straight_line(points, 0, knees[0], t)
    #slope = (y[j]-y[knees[0]]) / (x[j]-x[knees[0]])
    #rankings.append(math.fabs(slope))
    #r2 = rdp.get_r2(x[j:knees[0]+1], y[j:knees[0]+1])
    #coef = linear_fit(x[j:knees[0]+1], y[j:knees[0]+1])
    #r2 = linear_r2(x[j:knees[0]+1], y[j:knees[0]+1], coef)
    #rankings.append(r2)
    # height of the segment
    
    #weights.append(d)

    #top = j
    j = knees[0]
    #d = math.fabs(y[top] - y[knees[0]])
    #distances.append(d)

    #peak = y[knees[0]]
    peak = np.max(y[knees])

    for i in range(1, len(knees)):
        #print('Slope {}'.format(i))
        #j = rdp.naive_straight_line(points, knees[0], knees[i], t)
        #logger.info('%s - %s', x[j:knees[i]+1], y[j:knees[i]+1]) 

        #slope = (y[j]-y[knees[i]]) / (x[j]-x[knees[i]])
        #rankings.append(math.fabs(slope))
        r2 = rdp.get_r2(x[j:knees[i]+1], y[j:knees[i]+1])
        #coef = linear_fit(x[j:knees[i]+1], y[j:knees[i]+1])
        #r2 = linear_r2(x[j:knees[i]+1], y[j:knees[i]+1], coef)
        rankings.append(r2)
        
        # height of the segment
        d = math.fabs(peak - y[knees[i]])
        weights.append(d)

    weights = np.array(weights)
    #logger.info('d = %s', weights)
    #rankings = (rankings - np.min(rankings))/np.ptp(rankings)
    #rankings = rankings / np.sum(rankings)
    
    #sum_weights = np.sum(weights)
    #if sum_weights != 0:
    #    weights = weights / sum_weights
    
    max_weights = np.max(weights)
    if max_weights != 0:
        weights = weights / max_weights
    
    #weights = (weights - np.min(weights))/np.ptp(weights)
    #logger.info('w = %s',weights)
    rankings = np.array(rankings)
    #logger.info('r2 = %s',rankings)
    rankings = np.insert(rankings * weights, 0, 0., axis=0)
    #logger.info('rank = %s',rankings)

    if relative:
        rankings = rank(rankings)
        # Min Max normalization
        rankings = (rankings - np.min(rankings))/np.ptp(rankings)
    else:
        # Standardization (Z-score Normalization)
        rankings = (rankings - np.mean(rankings))/np.std(rankings)

    return rankings'''


def cluster_ranking(points: np.ndarray, knees: np.ndarray, t: ClusterRanking = ClusterRanking.linear) -> np.ndarray:
    # trivial cases
    if len(knees) == 0:
        return np.array([])
    elif len(knees) == 1:
        return np.array([1.0])
    else:
        fit = [0]
        weights  = [0]

        x = points[:,0]
        y = points[:,1]

        j = knees[0]
        peak = np.max(y[knees])

        for i in range(1, len(knees)-1):
            # R2 score
            r2 = 0
            if t is ClusterRanking.linear:
                r2_left = rdp.get_r2(x[j:knees[i]+1], y[j:knees[i]+1])
                r2_right = rdp.get_r2(x[knees[i]:knees[-1]], y[knees[i]:knees[-1]])
                r2 = (r2_left + r2_right) / 2.0
            elif t is ClusterRanking.left:
                r2 = rdp.get_r2(x[j:knees[i]+1], y[j:knees[i]+1])
            else:
                r2 = rdp.get_r2(x[knees[i]:knees[-1]], y[knees[i]:knees[-1]])
            fit.append(r2)
            
            # height of the segment
            d = math.fabs(peak - y[knees[i]])
            weights.append(d)

        weights.append(0)
        weights = np.array(weights)
        fit.append(0)
        fit = np.array(fit)
        
        max_weights = np.max(weights)
        if max_weights != 0:
            weights = weights / max_weights
        
        rankings = fit * weights

        # Compute relative ranking
        rankings = rank(rankings)
        # Min Max normalization
        rankings = (rankings - np.min(rankings))/np.ptp(rankings)

        return rankings



def multi_slope_ranking(points: np.ndarray, knees: np.ndarray, ts=[0.7, 0.8, 0.9]) -> np.ndarray:
    rankings = []

    for t in ts:
        rankings.append(slope_ranking(points, knees, t, True))
    
    rankings = np.array(rankings)

    #logger.info('Ranking %s', rankings)
    #median_rankings = np.median(rankings, axis=0)
    #rankings = np.average(rankings, axis=0)
    rankings = np.min(rankings, axis=0)
    #rankings = average_rankings
    #logger.info('Ranking %s', rankings)

    #global_ranking = []

    rankings = rank(rankings)
    #logger.info('Ranking %s', rankings)
    rankings = (rankings - np.min(rankings))/np.ptp(rankings)

    return rankings
