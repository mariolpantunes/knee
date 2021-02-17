# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import numpy as np
import math


def single_linkage(points, d, t=0.2):
    #total_duration = points[-1,0] - points[0,0]
    #print('TD = {}'.format(total_duration))

    clusters = []
    cluster_index = -1
    inside_cluster = False

    for i in range (0, len(points)-1):
        distance = math.fabs(points[i][0]-points[i+1][0])/d
        if distance < t:
            print('Merge {} with {}'.format(i,i+1))
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