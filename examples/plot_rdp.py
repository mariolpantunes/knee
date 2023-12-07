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


import enum
import logging
import argparse
import numpy as np

import knee.rdp as rdp
import knee.linear_fit as lf
import knee.metrics as metrics
import knee.evaluation as evaluation


import exectime.timeit as timeit


import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main(args):
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()

    points = np.genfromtxt(args.i, delimiter=',')

    if points.ndim == 1:
        y = points
        x = np.arange(0, len(y))
        points = np.array([x,y]).T
    
    ti, std, rv = timeit.timeit(args.n, rdp.rdp, points, t=args.r, cost=args.c, distance=args.d)
    reduced, _ = rv
    
    space_saving = round((1.0-(len(reduced)/len(points)))*100.0, 2)
    logger.info('Number of data points after RDP: %s(%s %%)', len(reduced), space_saving)
    cost = evaluation.compute_global_rmse(points, reduced)
    mip = evaluation.mip(points, reduced)
    logger.info(f'mIP = {mip}')
    logger.info(f'Global RMSE cost = {cost}')
    logger.info(f'Time = {ti} ± {std}')
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats.print_stats()

    x = points[:, 0]
    y = points[:, 1]
    plt.plot(x, y)

    selected = points[reduced]
    x = selected[:, 0]
    y = selected[:, 1]

    plt.plot(x, y, marker='o', markersize=3)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RDP test application')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-c', type=metrics.Metrics, choices=list(metrics.Metrics), help='cost metric', default='rpd')
    parser.add_argument('-d', type=rdp.Distance, choices=list(rdp.Distance), help='distance metric', default='shortest')
    parser.add_argument('-r', type=float, help='RDP reconstruction threshold', default=0.01)
    parser.add_argument('-n', type=int, help='number of repetition (for timeit)', default=3)
    args = parser.parse_args()
    
    main(args)