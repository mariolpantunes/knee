# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import numpy as np
import logging
import knee.knee_ranking as ranking
import matplotlib.pyplot as plt


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


def filter_worst_knees(points, knees):
    filtered_knees = []

    filtered_knees.append(knees[0])
    h_min = points[knees[0]][1]

    for i in range(1, len(knees)):
        h = points[knees[i]][1]
        if h <= h_min:
            filtered_knees.append(knees[i])
            h_min = h

    return np.array(filtered_knees)


def filter_clustring(points, knees, clustering, t=0.01, method=ranking.ClusterRanking.left, plot=False):
    xpoints = points[:,0]
    ypoints = points[:,1]
    
    knee_points = points[knees]
    clusters = clustering(knee_points, t)
    max_cluster = clusters.max()
    filtered_knees = []
    for i in range(0, max_cluster+1):
        current_cluster = knees[clusters==i]

        if len(current_cluster) > 1:
            rankings = ranking.cluster_ranking(points, current_cluster, method)
            #logger.info('Rankings: %s', rankings)
            idx = np.argmax(rankings)
            best_knee = knees[clusters==i][idx]
        else:
            #logger.info('Rankings: [1.0]')
            best_knee = knees[clusters==i][0]
        filtered_knees.append(best_knee)

        #plot cluster
        if plot:
            plt.plot(xpoints, ypoints)
            plt.plot(xpoints[current_cluster], ypoints[current_cluster], marker='x', markersize=3, color='green')
            plt.plot(xpoints[best_knee], ypoints[best_knee], marker='o', markersize=5, color='red')
            plt.show()
    return np.array(filtered_knees)