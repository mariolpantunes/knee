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
    
    points_reduced, removed = rdp.rdp(points, args.r)
    
    space_saving = round((1.0-(len(points_reduced)/len(points)))*100.0, 2)
    logger.info('Number of data points after RDP: %s(%s %%)', len(points_reduced), space_saving)
    
    indexes = np.arange(0, len(points_reduced))
    indexes = rdp.mapping(indexes, points_reduced, removed)

    hull_imp = {ConvexHull.hull: ch.graham_scan, ConvexHull.upper: ch.graham_scan_upper, ConvexHull.lower: ch.graham_scan_lower}

    if args.s is ConvexHullSource.raw:
        hull = hull_imp[args.c](points)
        hull_points = points[hull]
    else:
        hull = hull_imp[args.c](points_reduced)
        hull_points = points_reduced[hull]

    logger.info(hull)
    
    x = points[:, 0]
    y = points[:, 1]
    plt.plot(x, y)

    selected = points[indexes]
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
    parser.add_argument('-c', type=ConvexHull, choices=list(ConvexHull), default='lower')
    parser.add_argument('-r', type=float, help='RDP R2', default=0.99)
    args = parser.parse_args()
    
    main(args)