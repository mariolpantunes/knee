#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''


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

import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def kneedle_novel(points, args):
    reduced, removed = rdp.mp_grdp(points, t=args.r, min_points=args.p)

    points_reduced = points[reduced]
    knees = kneedle.auto_knees(points_reduced, p=kneedle.PeakDetection.All)

    knees = pp.filter_worst_knees(points_reduced, knees)
    knees = pp.filter_corner_knees(points_reduced, knees, t=args.c)
    knees = pp.filter_clusters(points_reduced, knees, clustering.average_linkage, args.t, args.k)
    knees = rdp.mapping(knees, reduced, removed)
    return knees


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

    # Plot knees
    x = points[:,0]
    y = points[:,1]
    plt.plot(x, y)

    # Novel Kneedle
    knees_02 = kneedle_novel(points, args)

    print(knees_02)
    
    plt.plot(x[knees_02], y[knees_02], 'r+')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kneedle evalution app')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-a', help='add even spaced points', action='store_true')
    #parser.add_argument('-s', type=float, help='sensitivity', default=1.0)
    #parser.add_argument('-tau', type=float, help='ema tau', default=1.0)
    
    parser.add_argument('-r', type=float, help='RDP reconstruction threshold', default=0.001)
    parser.add_argument('-p', type=int, help='minimum number of points', default=50)
    parser.add_argument('-t', type=float, help='clustering threshold', default=0.05)
    parser.add_argument('-c', type=float, help='corner threshold', default=0.33)
    parser.add_argument('-k', help='Knee ranking method', type=knee_ranking.ClusterRanking, choices=list(knee_ranking.ClusterRanking), default='left')
    args = parser.parse_args()

    main(args)

