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
import re
import csv
import enum
import math
import logging
import argparse
import numpy as np

import knee.rdp as rdp
import knee.kneedle as kneedle
import knee.postprocessing as pp
import knee.clustering as clustering
import knee.knee_ranking as knee_ranking


class Trace(enum.Enum):
    lru = 'lru'
    arc = 'arc'
    all = 'all'

    def __str__(self):
        return self.value


logging.basicConfig(level=logging.INFO, format='%(message)s')
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
    knees = pp.filter_worst_knees(points_reduced, knees)
    knees = pp.filter_corner_knees(points_reduced, knees, t=args.c)
    knees = pp.filter_clustring(points_reduced, knees, clustering.average_linkage, args.t, args.k)
    knees = rdp.mapping(knees, reduced, removed)
    return knees


def main(args):

    path = os.path.expanduser(args.p)

    if args.tr is Trace.all:
        files = [f for f in os.listdir(path) if re.match(r'w[0-9]*-(lru|arc)\.csv', f)]
    elif args.tr is Trace.arc:
        files = [f for f in os.listdir(path) if re.match(r'w[0-9]*-arc\.csv', f)]
    else:
        files = [f for f in os.listdir(path) if re.match(r'w[0-9]*-lru\.csv', f)]
    
    rv = []

    for i in range (len(files)):
        points = np.genfromtxt(f'{path}{files[i]}', delimiter=',')
        # open expected file
        dirname = os.path.dirname(f'{path}{files[i]}')
        filename = os.path.splitext(os.path.basename(files[i]))[0]
        expected_file = os.path.join(os.path.normpath(dirname), f'{filename}_expected.csv')
        expected = None
        if os.path.exists(expected_file):
            with open(expected_file, 'r') as f:
                reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
                expected = list(reader)
        else:
            expected = []
        expected = np.array(expected)
        # compute classical kneedle
        knees_01 = kneedle_classic(points, args, -1)
        #cm = evaluation.cm(points, knees_01, expected)
        #mcc01 = evaluation.mcc(cm)
        # compute novel kneedle
        knees_02 = kneedle_novel(points, args)
        #if len(knees_02) > 0:
        #    cm = evaluation.cm(points, knees_02, expected)
        #    mcc02 = evaluation.mcc(cm)
        #else:
        #    mcc02 = 0.0
        
        #logger.info(f'{mcc01:10.2E} {mcc02:10.2E}')
        if len(knees_01) > len(knees_02):
            solution = math.fabs(len(knees_01)-len(knees_02))
            rv.append((solution, i))
    rv.sort(reverse=True, key=lambda t: t[0])
    for solution, idx in rv: 
        logger.info(f'{files[idx]}({idx}): {solution}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find worst MRC trace')
    parser.add_argument('-p', type=str, help='input path', default='~/mrcs/')
    parser.add_argument('-tr', type=Trace, choices=list(Trace), default='arc')
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