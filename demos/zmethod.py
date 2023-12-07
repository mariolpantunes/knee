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
import matplotlib.pyplot as plt


import knee.rdp as rdp
import knee.linear_fit as lf
import knee.zmethod as zmethod
import knee.postprocessing as pp
import knee.evaluation as evaluation


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

    # get original x_max and y_ranges
    x_max = [max(x) for x in zip(*points)][0]
    y_range = [[max(y),min(y)] for y in zip(*points)][1]

    # run rdp
    reduced, removed = rdp.rdp(points, args.r)
    points_reduced = points[reduced]

    ## Knee detection code ##
    knees = zmethod.knees(points_reduced, dx=args.x, dy=args.y, dz=args.z, x_max=x_max, y_range=y_range)
    knees = knees[knees>0]

    ##########################

    # add even points
    if args.a:
        knees = pp.add_points_even(points, reduced, knees, removed)
    else:
        knees = rdp.mapping(knees, reduced, removed)

    rmspe_k = evaluation.rmspe(points, knees, expected, evaluation.Strategy.knees)
    rmspe_e = evaluation.rmspe(points, knees, expected, evaluation.Strategy.expected)
    cm = evaluation.cm(points, knees, expected, t = 0.01)
    mcc = evaluation.mcc(cm)

    logger.info(f'RMSE(knees)  RMSE(exp)  MCC')
    logger.info(f'-------------------------------------------')
    logger.info(f'{rmspe_k:10.2E} {rmspe_e:10.2E} {mcc:10.2E}')

    # store outpout
    if args.o:
        dirname = os.path.dirname(args.i)
        filename = os.path.splitext(os.path.basename(args.i))[0]
        output = os.path.join(os.path.normpath(dirname), f'{filename}_output.csv')

        dataset = points[knees]

        with open(output, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(dataset)
    
    # display result
    if args.g:
        x = points[:, 0]
        y = points[:, 1]
        plt.plot(x, y)
        plt.plot(x[knees], y[knees], 'r+')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi Knee evaluation app')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-a', help='add even spaced points', action='store_true')
    parser.add_argument('-c', type=lf.Linear_Metrics, choices=list(lf.Linear_Metrics), default='rpd')
    parser.add_argument('-d', type=rdp.RDP_Distance, choices=list(rdp.RDP_Distance), default='shortest')
    parser.add_argument('-r', type=float, help='RDP reconstruction threshold', default=0.01)
    parser.add_argument('-x', type=float, help='Parameter dx', default=0.01)
    parser.add_argument('-y', type=float, help='Parameter dy', default=0.01)
    parser.add_argument('-z', type=float, help='Parameter dz', default=0.5)
    parser.add_argument('-o', help='store output (debug)', action='store_true')
    parser.add_argument('-g', help='display output (debug)', action='store_true')
    args = parser.parse_args()

    main(args)
