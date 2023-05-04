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

import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def kneedle_classic(points, args, sw=-1):
    # For all possible sliding windows
    left = 0
    right = 0
    knees = []
    while right < len(points):
        if sw == -1:
            right = len(points)
        else:
            right = min(left+sw, len(points))
        #logger.info(f'[{left}, {right}]')
        window_points = points[left:right+1]
        window_knees = kneedle.knees(window_points, args.tau, args.cd, args.cc, args.s, debug=False)
        window_knees += left
        left = left + args.so
        knees.extend(window_knees.tolist())
    knees = np.unique(np.array(knees))
    return knees


def kneedle_novel(points, args):
    reduced, removed = rdp.rdp(points, args.r)
    points_reduced = points[reduced]
    knees = kneedle.auto_knees(points_reduced, p=kneedle.PeakDetection.All)
    
    #x = points_reduced[:, 0]
    #y = points_reduced[:, 1]
    #plt.plot(x, y)
    #plt.plot(x[knees], y[knees], 'r+')
    #plt.show()

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

    # Kneedle classic window
    #knees_00 = kneedle_classic(points, args, args.sw)
    #cm = evaluation.cm(points, knees_00, expected)
    #mcc00 = evaluation.mcc(cm)

    # Kneedle classic all
    knees_01 = kneedle_classic(points, args, -1)
    #cm = evaluation.cm(points, knees_01, expected)
    #mcc01 = evaluation.mcc(cm)

    # Novel Kneedle
    knees_02 = kneedle_novel(points, args)
    #if len(knees_02) > 0:
        #cm = evaluation.cm(points, knees_02, expected)
        #mcc02 = evaluation.mcc(cm)
    #else:
        #mcc02 = 0.0

    #logger.info(f'{mcc00:10.2E} {mcc01:10.2E} {mcc02:10.2E}')

    # Plot knees
    x = points[:,0]
    y = points[:,1]
    _, (ax2, ax3) = plt.subplots(1, 2)
    #ax1.plot(x, y)
    #ax1.plot(x[knees_00], y[knees_00], 'r+')
    ax2.plot(x, y)
    ax2.plot(x[knees_01], y[knees_01], 'r+')
    ax3.plot(x, y)
    if len(knees_02) > 0:
        ax3.plot(x[knees_02], y[knees_02], 'r+')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kneedle evalution app')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-a', help='add even spaced points', action='store_true')
    parser.add_argument('-s', type=float, help='sensitivity', default=1.0)
    parser.add_argument('-tau', type=float, help='ema tau', default=1.0)
    parser.add_argument('-cc', help='Rotation of a concavity', type=kneedle.Concavity, choices=list(kneedle.Concavity), default='counter-clockwise')
    parser.add_argument('-cd', help='Direction of a concavity', type=kneedle.Direction, choices=list(kneedle.Concavity), default='decreasing')
    parser.add_argument('-sw', help='Sliding window width', type=int, default=5000)
    parser.add_argument('-so', help='Sliding window overlap', type=int, default=1000)
    parser.add_argument('-r', type=float, help='RDP reconstruction threshold', default=0.001)
    parser.add_argument('-t', type=float, help='clustering threshold', default=0.05)
    parser.add_argument('-c', type=float, help='corner threshold', default=0.33)
    parser.add_argument('-k', help='Knee ranking method', type=knee_ranking.ClusterRanking, choices=list(knee_ranking.ClusterRanking), default='left')
    args = parser.parse_args()

    main(args)

