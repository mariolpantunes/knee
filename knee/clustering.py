# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import logging
import numpy as np


logger = logging.getLogger(__name__)


def single_linkage(points, t=0.2):
    clusters = []
    cluster_index = -1
    inside_cluster = False
    duration = points[-1,0] - points[0,0]

    for i in range (0, len(points)-1):
        distance = math.fabs(points[i][0]-points[i+1][0])/duration
        if distance < t:
            
            if not inside_cluster:
                inside_cluster = True
                cluster_index += 1
                clusters.append(cluster_index)
                clusters.append(cluster_index)
            else:
                clusters.append(cluster_index)
        else:
            if not inside_cluster:
                cluster_index += 1
                clusters.append(cluster_index)
            inside_cluster = False

    if not inside_cluster:
        cluster_index += 1
        clusters.append(cluster_index)

    return np.array(clusters)


def complete_linkage(points, t=0.01):
    clusters = []
    cluster_index = 0
    duration = points[-1,0] - points[0,0]

    # First Point is a cluster
    clusters.append(cluster_index)
    cluster_point_idx = 0

    for i in range (1, len(points)):
        distance = math.fabs(points[i][0]-points[cluster_point_idx][0])/duration
        if distance < t:
            clusters.append(cluster_index)
        else:
            cluster_index += 1
            clusters.append(cluster_index)
            cluster_point_idx = i

    return np.array(clusters)


def average_linkage(points, t=0.01):
    clusters = []
    cluster_index = 0
    duration = points[-1,0] - points[0,0]

    # First Point is a cluster
    clusters.append(cluster_index)
    cluster_center = points[0, 0]
    cluster_size = 1

    #logger.info('Cluster Center %s (%s)', cluster_center, cluster_size)

    for i in range (1, len(points)):
        distance = math.fabs(points[i][0] - cluster_center)/duration
        if distance < t:
            clusters.append(cluster_index)
            # Update center
            cluster_center = (cluster_size/(cluster_size+1)) * cluster_center + (1/(cluster_size+1)) * points[i][0]
            cluster_size += 1
            #logger.info('Cluster Center %s (%s)', cluster_center, cluster_size)
        else:
            cluster_index += 1
            clusters.append(cluster_index)
            cluster_center = points[i, 0]
            cluster_size = 1
            #logger.info('Cluster Center %s (%s)', cluster_center, cluster_size)

    return np.array(clusters)
