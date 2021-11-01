# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import argparse
import numpy as np
import logging

import matplotlib.pyplot as plt
import knee.rdp as rdp
import knee.convex_hull as ch


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main(args):
    points = np.genfromtxt(args.i, delimiter=',')
    
    points_reduced, removed = rdp.rdp(points, args.r)
    
    space_saving = round((1.0-(len(points_reduced)/len(points)))*100.0, 2)
    logger.info('Number of data points after RDP: %s(%s %%)', len(points_reduced), space_saving)
    
    indexes = np.arange(0, len(points_reduced))
    indexes = rdp.mapping(indexes, points_reduced, removed)

    #hull = ch.graham_scan(points_reduced)
    hull = ch.graham_scan(points)

    logger.info(hull)
    
    x = points[:, 0]
    y = points[:, 1]
    plt.plot(x, y)

    selected = points[indexes]
    x = selected[:, 0]
    y = selected[:, 1]

    plt.plot(x, y, marker='o', markersize=3)
    
    #for p in hull:
    #    print(p)
    #    plt.plot(p[0], p[1], 'c')
    x = hull[:, 0]
    y = hull[:, 1]
    plt.plot(x, y, 'o', mec='r', color='none', lw=1, markersize=10)
    plt.fill(x, y, edgecolor='r', fill=False)
    
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RDP test application')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-r', type=float, help='RDP R2', default=0.9)
    args = parser.parse_args()
    
    main(args)