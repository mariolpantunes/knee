# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


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
    points = np.genfromtxt(args.i, delimiter=',')
    logger.info(f'Distance {args.d} Order {args.o}')
    
    reduced, _ = rdp.rdp_fixed(points, args.l, distance=args.d, )
    
    ti, std, rv = timeit.timeit(args.n, rdp.rdp_fixed, points, length=args.l, distance=args.d, order=args.o)
    reduced, _ = rv
    
    space_saving = round((1.0-(len(reduced)/len(points)))*100.0, 2)
    logger.info('Number of data points after RDP: %s(%s %%)', len(reduced), space_saving)
    cost = evaluation.compute_global_rmse(points, reduced)
    mip = evaluation.mip(points, reduced)
    logger.info(f'mIP = {mip}')
    logger.info(f'Global RMSE cost = {cost}')
    logger.info(f'Time = {ti} ± {std}')
    
    x = points[:, 0]
    y = points[:, 1]
    plt.plot(x, y)

    points_reduced = points[reduced]
    x = points_reduced[:, 0]
    y = points_reduced[:, 1]
    plt.plot(x, y, marker='o', markersize=3)
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RDP (fixed length) test application')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-l', type=int, help='number of points', default=10)
    parser.add_argument('-d', type=rdp.Distance, choices=list(rdp.Distance), help='distance metric', default='shortest')
    parser.add_argument('-o', type=rdp.Order, choices=list(rdp.Order), help='ordering metric', default='area')
    parser.add_argument('-n', type=int, help='number of repetition (for timeit)', default=3)
    args = parser.parse_args()
    
    main(args)