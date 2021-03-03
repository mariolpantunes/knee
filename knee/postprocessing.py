# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import numpy as np
import logging
import clustering
import ranking


logger = logging.getLogger(__name__)


def corner_point(points, knee, t=0.01):
    x = points[:,0]
    y = points[:,1]

    lenght = x[-1] - x[0]
    fxp = y[knee]
    i = knee + 1
    done = False
    rv = True

    while not done and rv:
        if y[i] > fxp:
            rv = False
        
        p = (x[i] - x[i])/lenght
        if p >= t:
            done = True
        
        i += 1
    return rv


def filter_corner_point(points, knees):
    logger.info('Filter Corner Points')

    rv = []

    for knee in knees:
        if not corner_point(points, knee):
            rv.append(knee)
    
    return np.array(rv)


def filter_clustring(points, knees, t=0.01):
    logger.info('Filter Knees based on 1D Clustering and Ranking')

    knee_points = points[knees]
    clusters = clustering.single_linkage(knee_points, t)
    max_cluster = clusters.max()
    filtered_knees = []
    for i in range(0, max_cluster+1):
        print('Cluster {}'.format(i))
        current_cluster = knees[clusters==i]
        print(current_cluster)
        if len(current_cluster) > 1:
            rankings = ranking.slope_ranking(points, current_cluster)
            print(rankings)
            idx = np.argmax(rankings)
            filtered_knees.append(knees[clusters==i][idx])
        else:
            filtered_knees.append(knees[clusters==i][0])


def postprocessing(points, knees, t=0.01):
    
    keys = ['knees', 'knees_z', 'knees_significant', 'knees_iso']

    

    rv = knees.copy()

    for k in keys:
        
        current_knees = knees[k]
        #print(current_knees)
        knee_points = points[current_knees]
        #print(knee_points)
        
        
        
        
        filtered_knees = []
        for i in range(0, max_cluster+1):
            print('Cluster {}'.format(i))
            current_cluster = current_knees[clusters==i]
            print(current_cluster)
            if len(current_cluster) > 1:
                rankings = slope_ranking(points, current_cluster)
                print(rankings)
                idx = np.argmax(rankings)
                filtered_knees.append(current_knees[clusters==i][idx])
            else:
                filtered_knees.append(current_knees[clusters==i][0])
        rv[k] = np.array(filtered_knees)
    return rv