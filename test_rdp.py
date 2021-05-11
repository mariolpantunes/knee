# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import os
import argparse
import numpy as np
import logging

from enum import Enum

from knee.knee_ranking import rank, slope_ranking
from knee.postprocessing import filter_clustring, filter_worst_knees
import matplotlib.pyplot as plt
from knee.rdp import rdp, mapping
import knee.clustering as clustering
from plot import plot_ranking


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class Clustering(Enum):
    single = 'single'
    complete = 'complete'
    average = 'average'

    def __str__(self):
        return self.value


def postprocessing(points, knees, args):
    logger.info('Post Processing')
    logger.info('Knees: %s', knees)
    logger.info('Initial #Knees: %s', len(knees))
    knees = filter_worst_knees(points, knees)
    logger.info('Worst Knees: %s', len(knees))
    cmethod = {Clustering.single: clustering.single_linkage, Clustering.complete: clustering.complete_linkage, Clustering.average: clustering.average_linkage}
    current_knees = filter_clustring(points, knees, cmethod[args.c], args.t)
    logger.info('Clustering Knees: %s', len(current_knees))

    return current_knees


def main(args):
    points = np.genfromtxt(args.i, delimiter=',')
    points_reduced, removed = rdp(points, args.r)
    space_saving = round((1.0-(len(points_reduced)/len(points)))*100.0, 2)
    logger.info('Number of data points after RDP: %s(%s %%)', len(points_reduced), space_saving)
    
    indexes = np.arange(0, len(points_reduced))
    logger.info(indexes)

    indexes = mapping(indexes, points_reduced, removed)
    logger.info(indexes)

    x = points[:, 0]
    y = points[:, 1]
    plt.plot(x, y)

    selected = points[indexes]
    x = selected[:, 0]
    y = selected[:, 1]

    plt.plot(x, y, marker='o', markersize=3)

    #knees = np.arange(1, len(points_reduced))
    #filtered_knees = postprocessing(points_reduced, knees, args)
    
    #plot_ranking(plt, points_reduced, filtered_knees, slope_ranking, '')
    plt.show()
    #plt.savefig(args.o)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RDP test application')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-r', type=float, help='RDP R2', default=0.95)
    parser.add_argument('-c', type=Clustering, choices=list(Clustering), default='average')
    parser.add_argument('-t', type=float, help='clustering threshold', default=0.02)
    #parser.add_argument('-o', type=str, required=True, help='output file')
    args = parser.parse_args()
    
    main(args)