#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import os
import csv
import math
import argparse
import numpy as np
import logging


from enum import Enum
import knee.rdp as rdp
import knee.kneedle as kneedle
import knee.postprocessing as pp
import knee.clustering as clustering
import knee.evaluation as evaluation
from knee.knee_ranking import ClusterRanking


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class Clustering(Enum):
    single = 'single'
    complete = 'complete'
    average = 'average'

    def __str__(self):
        return self.value


def main(args):
    # define clustering methods
    cmethod = {Clustering.single: clustering.single_linkage, Clustering.complete: clustering.complete_linkage, Clustering.average: clustering.average_linkage}
    # get the expected file from the input file
    dirname = os.path.dirname(args.i)
    filename = os.path.splitext(os.path.basename(args.i))[0]
    expected_file = os.path.join(os.path.normpath(dirname), f'{filename}_expected.csv')

    expected = None

    if os.path.exists(expected_file):
        with open(expected_file, 'r') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            expected = list(reader)
    else:
        expected = []

    points = np.genfromtxt(args.i, delimiter=',')

    ## Knee detection code ##

    points_reduced, points_removed = rdp.rdp(points, args.r)
    knees = kneedle.auto_knees(points_reduced)
    t_k = pp.filter_worst_knees(points_reduced, knees)
    t_k = pp.filter_corner_knees(points_reduced, t_k)
    filtered_knees = pp.filter_clustring(points_reduced, t_k, cmethod[args.c], args.t, args.m)
    if args.a:
        knees = pp.add_points_even(points, points_reduced, filtered_knees, points_removed)
    else:
        knees = rdp.mapping(filtered_knees, points_reduced, points_removed)
    
    ##########################

    logger.info(f'MSE(knees)   MSE(exp)   Cost(tr)   Cost(kn)')
    logger.info(f'-------------------------------------------')
    if len(expected) > 0:
        error_mse = evaluation.mse(points, knees, expected, evaluation.Strategy.knees)
        error_mse_exp = evaluation.mse(points, knees, expected, evaluation.Strategy.expected)
    else:
        error_mse = math.nan
        error_mse_exp = math.nan
    _,_,_,_,cost_trace = evaluation.accuracy_trace (points, knees)
    _,_,_,_,cost_knee = evaluation.accuracy_knee (points, knees)
    logger.info(f'{error_mse:10.2E} {error_mse_exp:10.2E} {cost_trace:10.2E} {cost_knee:10.2E}')

    # store outpout
    if args.o:
        dirname = os.path.dirname(args.i)
        filename = os.path.splitext(os.path.basename(args.i))[0]
        output = os.path.join(os.path.normpath(dirname), f'{filename}_output.csv')

        dataset = points[knees]

        with open(output, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi Knee evaluation app')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-r', type=float, help='RDP R2', default=0.95)
    parser.add_argument('-c', type=Clustering, choices=list(Clustering), help='clustering metric', default='average')
    parser.add_argument('-t', type=float, help='clustering threshold', default=0.05)
    parser.add_argument('-m', type=ClusterRanking, choices=list(ClusterRanking), help='direction of the cluster ranking', default='left')
    parser.add_argument('-a', help='add even spaced points', action='store_true')
    parser.add_argument('-o', help='store output (debug)', action='store_true')
    args = parser.parse_args()
    
    main(args)
