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
import knee.zmethod as zmethod

import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
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

    # Plot knees
    x = points[:,0]
    y = points[:,1]
    plt.plot(x, y)

    # Z-method
    reduced, removed = rdp.mp_grdp(points, t=args.r, min_points=args.p)
    points_reduced = points[reduced]

    knees = zmethod.knees(points_reduced, args.dx, args.dy, args.dz, x_max=max(x), y_range=[max(y),min(y)])
    #knees = zmethod.knees(points, args.dx, args.dy, args.dz, x_max=max(x), y_range=[max(y),min(y)])
    knees = knees[knees > 0]
    knees = rdp.mapping(knees, reduced, removed)

    logger.info(f'Number of knees {len(knees)}')
    plt.plot(x[knees], y[knees], 'r+')

    knees = zmethod.knees2(points_reduced, args.dx, args.dy)
    knees = rdp.mapping(knees, reduced, removed)
    logger.info(f'Number of knees {len(knees)}')
    plt.plot(x[knees], y[knees], 'g+')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kneedle evalution app')
    parser.add_argument('-i', type=str, required=True, help='input file')

    parser.add_argument('-r', type=float, help='RDP reconstruction threshold', default=0.01)
    parser.add_argument('-p', type=int, help='minimum number of points', default=50)
    parser.add_argument('--dx', type=float, help='minimum number of points', default=.05)
    parser.add_argument('--dy', type=float, help='minimum number of points', default=.05)
    parser.add_argument('--dz', type=float, help='minimum number of points', default=.05)
    parser.add_argument('-o', help='Outlier detection method', type=zmethod.Outlier, choices=list(zmethod.Outlier), default='iqr')

    #parser.add_argument('-a', help='add even spaced points', action='store_true')
    
    args = parser.parse_args()

    main(args)

