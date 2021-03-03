# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import numpy as np
import logging
import knee.knee_ranking as ranking


logger = logging.getLogger(__name__)


def corner_point(points, knee, t=0.01):
    x = points[:,0]
    y = points[:,1]

    lenght = x[-1] - x[0]
    fxp = y[knee]
    i = knee + 1
    done = False
    rv = True

    #logger.info('Is corner Point (%s): %s', i-1, rv)

    while not done and rv:
        if y[i] >= fxp:
            rv = False
        #logger.info('Is corner Point (%s): %s', i, rv)
        
        p = (x[i] - x[knee])/lenght
        if p >= t:
            done = True
        
        i += 1
        if i > len(points) - 1: 
            done = True
    #logger.info('(Last) Is corner Point (%s): %s', i, rv)

    return rv


def filter_corner_point(points, knees, t=0.01):
    logger.info('Filter Corner Points')

    rv = []

    for knee in knees:
        if not corner_point(points, knee, t):
            rv.append(knee)
    
    return np.array(rv)


def filter_clustring(points, knees, clustering, t=0.01):
    logger.info('Filter Knees based on 1D Clustering and Ranking')

    knee_points = points[knees]
    clusters = clustering(knee_points, t)
    max_cluster = clusters.max()
    filtered_knees = []
    for i in range(0, max_cluster+1):
        current_cluster = knees[clusters==i]
        if len(current_cluster) > 1:
            rankings = ranking.slope_ranking(points, current_cluster)
            idx = np.argmax(rankings)
            filtered_knees.append(knees[clusters==i][idx])
        else:
            filtered_knees.append(knees[clusters==i][0])
    return np.array(filtered_knees)