#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import os
import csv
import logging
import argparse
import numpy as np


import knee.rdp as rdp
import knee.kneedle as kneedle
import knee.postprocessing as pp
import knee.clustering as clustering
import knee.evaluation as evaluation
import knee.knee_ranking as knee_ranking


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main(args):
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
    expected = np.array(expected)
    points = np.genfromtxt(args.i, delimiter=',')

    ## Knee detection code ##

    points_reduced, points_removed = rdp.rdp(points, args.r)
    knees = kneedle.auto_knees(points_reduced)
    t_k = pp.filter_worst_knees(points_reduced, knees)
    t_k = pp.filter_corner_knees(points_reduced, t_k, t=args.c)
    filtered_knees = pp.filter_clustring(points_reduced, t_k, clustering.average_linkage, args.t, knee_ranking.ClusterRanking.left)
    
    ##########################################################################################
    
    # add even points
    if args.a:
        knees = pp.add_points_even(points, points_reduced, filtered_knees, points_removed, 0.009, 0.009)
    else:
        knees = rdp.mapping(filtered_knees, points_reduced, points_removed)

    nk = len(knees)

    if nk > 0:
        rmspe_k = evaluation.rmspe(points, knees, expected, evaluation.Strategy.knees)
        rmspe_e = evaluation.rmspe(points, knees, expected, evaluation.Strategy.expected)
        cm = evaluation.cm(points, knees, expected, t = 0.01)
        mcc = evaluation.mcc(cm)
    else:
        rmspe_k = 999
        rmspe_e = 999
        mcc = -1

    logger.info(f'RMSE(knees)  RMSE(exp)  MCC    N_Knees')
    logger.info(f'-------------------------------------------')
    logger.info(f'{rmspe_k:10.2E} {rmspe_e:10.2E} {mcc:10.2E}  {nk}')

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
    parser.add_argument('-a', help='add even spaced points', action='store_true')
    parser.add_argument('-r', type=float, help='RDP R2', default=0.93)
    parser.add_argument('-t', type=float, help='clustering threshold', default=0.06)
    parser.add_argument('-c', type=float, help='corner threshold', default=0.41)
    parser.add_argument('-o', help='store output (debug)', action='store_true')
    args = parser.parse_args()
    
    main(args)
