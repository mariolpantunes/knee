#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import os
import csv
import argparse
import numpy as np
import logging


from enum import Enum
import knee.rdp as rdp
import knee.kneedle as kneedle
import knee.lmethod as lmethod
import knee.dfdt as dfdt
import knee.menger as menger
import knee.curvature as curvature
import knee.postprocessing as pp
import matplotlib.pyplot as plt
import knee.clustering as clustering
import knee.pointSelector as ps
import knee.knee_ranking as ranking
import knee.evaluation as evaluation
from knee.knee_ranking import ClusterRanking
from plot import get_dimention, plot_lines_knees_ranking, plot_lines_knees


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class Clustering(Enum):
    single = 'single'
    complete = 'complete'
    average = 'average'

    def __str__(self):
        return self.value


def plot_knees_ranking(points, knees, names, rankings, expected):
    x = points[:,0]
    y = points[:,1]

    nrows = ncols = 1
    
    if len(knees) > 1:
        nrows, ncols = get_dimention(len(knees))
    
    _, axs = plt.subplots(nrows, ncols)

    for j in range(len(knees)):
        if len(knees) == 1:
            plot_lines_knees_ranking(axs, x, y, knees[j], rankings[j], names[j])
        elif nrows == 1:
            plot_lines_knees_ranking(axs[j], x, y, knees[j], rankings[j], names[j])
        else:
            c = j % ncols
            r = j // ncols
            plot_lines_knees_ranking(axs[r, c], x, y, knees[j], rankings[j], names[j])
        

    #filename = os.path.splitext(args.i)[0]+'_ranking.pdf'
    #plt.savefig(filename, transparent = True, bbox_inches = 'tight', pad_inches = 0, dpi = 300)
    #fig.tight_layout()
    plt.figure(300)
    plt.plot(x, y)
    for x,y in expected:
        plt.plot(x, y, marker='o', markersize=3, color='red')
    plt.show()



def plot_knees(points, knees, names):
    x = points[:,0]
    y = points[:,1]

    nrows = ncols = 1
    
    if len(names) > 1:
        nrows, ncols = get_dimention(len(names))
    
    _, axs = plt.subplots(nrows, ncols)

    for j in range(len(names)):
        
        if len(knees) == 1:
            plot_lines_knees(axs, x, y, knees[j], names[j])
        elif nrows == 1:
            plot_lines_knees(axs[j], x, y, knees[j], names[j])
        else:
            c = j % ncols
            r = j // ncols
            plot_lines_knees(axs[r, c], x, y, knees[j], names[j])
        

    #filename = os.path.splitext(args.i)[0]+'_ranking.pdf'
    #plt.savefig(filename, transparent = True, bbox_inches = 'tight', pad_inches = 0, dpi = 300)
    #fig.tight_layout()
    plt.show()


def main(args):
    # get the expected file from the input file
    dirname = os.path.dirname(args.i)
    output = os.path.join(os.path.normpath(dirname), 'expected.csv')

    expected = None

    if os.path.exists(output):
        with open(output, 'r') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            expected = list(reader)
    else:
        expected = []

    points = np.genfromtxt(args.i, delimiter=',')
    points_reduced, points_removed = rdp.rdp(points, args.r)

    space_saving = round((1.0-(len(points_reduced)/len(points)))*100.0, 2)
    logger.info('Number of data points after RDP: %s(%s %%)', len(points_reduced), space_saving)

    names = ['kneedle', 'kneedke(Rec)', 'l-method', 'dfdt', 'menger', 'curvature', 'Tyler (RDP)', 'Tyler', 'RDP'] 
    methods = [kneedle.auto_knees, kneedle.multi_knee, lmethod.multi_knee, dfdt.multi_knee, menger.multi_knee, curvature.multi_knee, ps.get_knees_points]
    knees = []
    knees_raw = []

    # Elbow methods
    for m, n in zip(methods, names):
        tmp = m(points_reduced)
        knees.append(tmp)
        raw_indexes = rdp.mapping(tmp, points_reduced, points_removed)
        knees_raw.append(raw_indexes)
    
    # Tyler
    candidates = ps.get_knees_points(points)
    knees.append(candidates)
    knees_raw.append(candidates)
    
    # RDP
    candidates = np.arange(1, len(points_reduced))
    knees.append(candidates)
    raw_indexes = rdp.mapping(candidates, points_reduced, points_removed)
    knees_raw.append(raw_indexes)

    plot_knees(points, knees_raw, names)

    cmethod = {Clustering.single: clustering.single_linkage, Clustering.complete: clustering.complete_linkage, Clustering.average: clustering.average_linkage}

    # Cluster and select points
    filtered_knees_raw = []
    rankings = []
    for k, n in zip(knees, names):
        if n == 'Tyler':
            filtered_knees_raw.append(k)
            ranks = np.full(len(k), 1.0)
            #rankings.append(ranking.slope_ranking(points, k))
            rankings.append(ranks)
        else:
            t_k = pp.filter_worst_knees(points_reduced, k)
            filtered_knees = pp.filter_clustring(points_reduced, t_k, cmethod[args.c], args.t, args.m)
            rankings.append(ranking.slope_ranking(points_reduced, filtered_knees))
            raw_indexes = rdp.mapping(filtered_knees, points_reduced, points_removed)
            filtered_knees_raw.append(raw_indexes)
    
    logger.info(f'Model         MSE      MSE(exp) Cost(tr) Cost(kn)')
    for k, n in zip(filtered_knees_raw, names):
        error_mse = evaluation.mse(points, k, expected)
        error_mse_exp = evaluation.mse(points, k, expected, True)
        _,_,_,_,cost_trace = evaluation.accuracy_trace (points, k)
        _,_,_,_,cost_knee = evaluation.accuracy_knee (points, k)
        logger.info(f'{n:<13} {error_mse:.2E} {error_mse_exp:.2E} {cost_trace:.2E} {cost_knee:.2E}')

    plot_knees_ranking(points, filtered_knees_raw, names, rankings, expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi Knee evaluation app')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-r', type=float, help='RDP R2', default=0.9)
    parser.add_argument('-c', type=Clustering, choices=list(Clustering), default='average')
    parser.add_argument('-t', type=float, help='clustering threshold', default=0.02)
    parser.add_argument('-m', type=ClusterRanking, choices=list(ClusterRanking), default='left')
    args = parser.parse_args()
    
    main(args)
