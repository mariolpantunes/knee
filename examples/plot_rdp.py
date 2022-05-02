# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import enum
import logging
import argparse
import numpy as np

import knee.rdp as rdp
import knee.linear_fit as lf
import knee.metrics as metrics
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main(args):
    points = np.genfromtxt(args.i, delimiter=',')

    if points.ndim == 1:
        y = points
        x = np.arange(0, len(y))
        points = np.array([x,y]).T
    
    reduced, removed = rdp.rdp(points, args.r, cost=args.c, distance=args.d)
    
    space_saving = round((1.0-(len(reduced)/len(points)))*100.0, 2)
    logger.info('Number of data points after RDP: %s(%s %%)', len(reduced), space_saving)
    cost, _ = rdp.compute_global_cost(points, reduced, cost=args.c)
    logger.info(f'Global cost = {cost}')
    
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
    parser.add_argument('-c', type=metrics.Metrics, choices=list(metrics.Metrics), default='rpd')
    parser.add_argument('-d', type=rdp.Distance, choices=list(rdp.Distance), default='shortest')
    parser.add_argument('-r', type=float, help='RDP R', default=0.01)
    args = parser.parse_args()
    
    main(args)