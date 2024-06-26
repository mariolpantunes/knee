#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '1.0'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''


import os
import csv
import argparse
import numpy as np
import logging


import kneeliverse.rdp as rdp
import kneeliverse.postprocessing as pp
import kneeliverse.clustering as clustering
import kneeliverse.evaluation as evaluation
import kneeliverse.knee_ranking as knee_ranking #import ClusterRanking


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

    rs = [0.75, 0.80, 0.85, 0.90, 0.95]
    ts = [0.01, 0.02, 0.03, 0.04, 0.05]

    evaluations = []

    for r in rs:
        ## Knee detection code ##
        points_reduced, points_removed = rdp.rdp(points, r)
        knees = np.arange(1, len(points_reduced))
        t_k = pp.filter_worst_knees(points_reduced, knees)
        t_k = pp.filter_corner_knees(points_reduced, t_k)
        for t in ts:
            ## Clustering ##
            filtered_knees = pp.filter_clustring(points_reduced, t_k, clustering.average_linkage, t, knee_ranking.ClusterRanking.left)
            final_knees = pp.add_points_even(points, points_reduced, filtered_knees, points_removed)
            
            ## Evaluation ##
            error_rmspe = evaluation.rmspe(points, final_knees, expected, evaluation.Strategy.knees)
            error_rmspe_exp = evaluation.rmspe(points, final_knees, expected, evaluation.Strategy.expected)

            _,_,_,_,cost_trace = evaluation.accuracy_trace (points, final_knees)
            _,_,_,_,cost_knee = evaluation.accuracy_knee (points, final_knees)
            
            evaluations.append([error_rmspe, error_rmspe_exp, cost_trace, cost_knee])
    
    ## Compute the Correlation ##
    evaluations = np.array(evaluations)
    rho = np.corrcoef(evaluations.T)
    rmspe_rmspe_exp = rho[0,1]
    rmspe_cost_trace = rho[0,2]
    rmspe_cost_knee = rho[0,3]

    rmspe_exp_cost_trace = rho[1,2]
    rmspe_exp_cost_knee = rho[1,3]

    cost_trace_cost_knee = rho[2,3]

    #logger.info(f'{rho}')
    logger.info(f'{rmspe_rmspe_exp}, {rmspe_cost_trace}, {rmspe_cost_knee}, {rmspe_exp_cost_trace}, {rmspe_exp_cost_knee}, {cost_trace_cost_knee}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi Knee evaluation app')
    parser.add_argument('-i', type=str, required=True, help='input file')
    args = parser.parse_args()
    
    main(args)