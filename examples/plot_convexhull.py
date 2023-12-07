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
import knee.convex_hull as ch
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class ConvexHullSource(enum.Enum):
    raw = 'raw'
    rdp = 'rdp'

    def __str__(self):
        return self.value

class ConvexHull(enum.Enum):
    hull = 'hull'
    lower = 'lower'
    upper = 'upper'

    def __str__(self):
        return self.value


def main(args):
    points = np.genfromtxt(args.i, delimiter=',')

    if points.ndim == 1:
        y = points
        x = np.arange(0, len(y))
        points = np.array([x,y]).T
    
    reduced, _ = rdp.rdp(points, args.r, cost=args.c, distance=args.d)
    
    space_saving = round((1.0-(len(reduced)/len(points)))*100.0, 2)
    logger.info('Number of data points after RDP: %s(%s %%)', len(reduced), space_saving)
    
    hull_imp = {ConvexHull.hull: ch.graham_scan, ConvexHull.upper: ch.graham_scan_upper, ConvexHull.lower: ch.graham_scan_lower}

    selected = points[reduced]

    if args.s is ConvexHullSource.raw:
        hull = hull_imp[args.ch](points)
        hull_points = points[hull]
    else:
        hull = hull_imp[args.ch](selected)
        hull_points = selected[hull]

    logger.info(hull)
    
    x = points[:, 0]
    y = points[:, 1]
    plt.plot(x, y)

    
    x = selected[:, 0]
    y = selected[:, 1]
    plt.plot(x, y, marker='o', markersize=3)
    
    x = hull_points[:, 0]
    y = hull_points[:, 1]
    plt.plot(x, y, 'o', mec='r', color='none', lw=1, markersize=10)
    plt.fill(x, y, edgecolor='r', fill=False)
    
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RDP test application')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-s', type=ConvexHullSource, choices=list(ConvexHullSource), default='rdp')
    parser.add_argument('-ch', type=ConvexHull, choices=list(ConvexHull), default='upper')
    parser.add_argument('-c', type=lf.Linear_Metrics, choices=list(lf.Linear_Metrics), default='rpd')
    parser.add_argument('-d', type=rdp.RDP_Distance, choices=list(rdp.RDP_Distance), default='shortest')
    parser.add_argument('-r', type=float, help='RDP R', default=0.01)
    args = parser.parse_args()
    
    main(args)