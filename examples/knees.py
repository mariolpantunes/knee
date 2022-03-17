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
import knee.postprocessing as pp
import knee.clustering as clustering
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
    
    points_reduced = points[reduced]

    # all rdp points are candidates, except extremes
    knees = np.arange(1, len(points_reduced))
    logger.info(f'Knees {len(knees)}')

    # filter out all non-corner points
    knees = pp.select_corner_knees(points_reduced, knees, t=args.t1)
    logger.info(f'Knees {len(knees)}')

    # cluster points together
    knees = pp.filter_clusters_corners(points_reduced, knees, clustering.average_linkage, t=args.t2)
    logger.info(f'Knees {len(knees)}')
    
    x = points[:, 0]
    y = points[:, 1]
    plt.plot(x, y)

    # map the points to the original space
    knees = rdp.mapping(knees, reduced, removed)
    
    #rdp_points = points[reduced]
    #x = rdp_points[:, 0]
    #y = rdp_points[:, 1]
    #plt.plot(x, y, marker='o', markersize=3, linestyle = 'None')

    knee_points = points[knees]
    x = knee_points[:, 0]
    y = knee_points[:, 1]
    plt.plot(x, y, marker='o', markersize=3, linestyle = 'None')
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RDP test application')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-c', type=lf.Linear_Metrics, choices=list(lf.Linear_Metrics), default='rpd')
    parser.add_argument('-d', type=rdp.RDP_Distance, choices=list(rdp.RDP_Distance), default='shortest')
    parser.add_argument('-t1', type=float, help='Corner Threshold', default=0.33)
    parser.add_argument('-t2', type=float, help='Clustering Threshold', default=0.05)
    parser.add_argument('-r', type=float, help='RDP R', default=0.01)
    args = parser.parse_args()
    
    main(args)