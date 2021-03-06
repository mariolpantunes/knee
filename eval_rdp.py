# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import argparse
import numpy as np
import logging

from enum import Enum
from knee.evaluation import accuracy_trace
from knee.knee_ranking import rank, slope_ranking
from knee.postprocessing import filter_clustring, filter_worst_knees, filter_corner_knees
import matplotlib.pyplot as plt
from knee.rdp import rdp, mapping
from knee.knee_ranking import ClusterRanking
import knee.clustering as clustering
from plot import plot_ranking, plot_knees


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
    #logger.info('Knees: %s', knees)
    logger.info('Initial #Knees: %s', len(knees))
    wknees = filter_worst_knees(points, knees)

    #plot_knees(plt, points, knees, 'Worst Knees')
    logger.info('After Worst #Knees: %s', len(knees))
    
    knees = filter_corner_knees(points, wknees)
    #plot_knees(plt, points, knees, 'Corner Knees')

    #diff = np.setdiff1d(wknees, knees)
    #plot_knees(plt, points, diff, 'Diff Knees')

    logger.info('After Corner #Knees: %s', len(knees))
    #logger.info('Worst Knees: %s', len(knees))
    cmethod = {Clustering.single: clustering.single_linkage, Clustering.complete:
               clustering.complete_linkage, Clustering.average: clustering.average_linkage}
    current_knees = filter_clustring(
        points, knees, cmethod[args.c], args.t, args.m)
    logger.info('Clustering Knees: %s', len(current_knees))

    return current_knees


def main(args):
    points = np.genfromtxt(args.i, delimiter=',')
    points_reduced, removed = rdp(points, args.r)
    #space_saving = round((1.0-(len(points_reduced)/len(points)))*100.0, 2)
    #logger.info('Number of data points after RDP: %s(%s %%)', len(points_reduced), space_saving)

    knees = np.arange(1, len(points_reduced))
    raw_knees = mapping(knees, points_reduced, removed)
    plot_knees(plt, points, raw_knees, 'Knees')
    #logger.info('Knee extraction')
    filtered_knees = postprocessing(points_reduced, knees, args)
    rankings = slope_ranking(points_reduced, filtered_knees)
    #logger.info('Clustering and ranking')
    filtered_knees = mapping(filtered_knees, points_reduced, removed)
    #logger.info('Mapping into raw plot')

    # Compute performance evalution
    average_x, average_y, average_slope, average_coeffients, cost = accuracy_trace(points, filtered_knees)
    logger.info('Performance %s %s %s %s %s', average_x, average_y, average_slope, average_coeffients, cost)

    plot_ranking(plt, points, filtered_knees, rankings, '')  # args.o)
    plt.show()
    # plt.savefig(args.o)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RDP test application')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-r', type=float, help='RDP R2', default=0.95)
    parser.add_argument('-c', type=Clustering,choices=list(Clustering), default='average')
    parser.add_argument('-t', type=float, help='clustering threshold', default=0.05)
    parser.add_argument('-m', type=ClusterRanking,choices=list(ClusterRanking), default='left')
    #parser.add_argument('-o', type=str, required=True, help='output file')
    args = parser.parse_args()

    main(args)
