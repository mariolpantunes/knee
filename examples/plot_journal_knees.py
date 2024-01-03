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

import tqdm
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt 

from sklearn import datasets
from sklearn.cluster import KMeans

import knee.rdp as rdp
import knee.kneedle as kneedle
import knee.postprocessing as pp
import knee.clustering as clustering
import knee.knee_ranking as knee_ranking


plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['lines.linewidth'] = 2


logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main(args):
    colormap=np.array(['#4C72B0','#DD8452','#55A868','#C44E52',
    '#8172B3','#937860','#DA8BC3','#8C8C8C','#CCB974','#64B5CD'])
    points = np.genfromtxt(args.i, delimiter=',')
    logger.info(f'Loading {args.i} file ({len(points)})...')
    reduced, removed = rdp.mp_grdp(points, t=0.001, min_points=20)
    space_saving = round((1.0-(len(reduced)/len(points)))*100.0, 2)
    logger.info('Number of data points after RDP: %s(%s %%)', len(reduced), space_saving)

    # Plot original trace and reduced version
    x = points[:,0]
    y = points[:,1]
    plt.plot(x, y, color= colormap[0])
    points_reduced = points[reduced]
    plt.plot(points_reduced[:, 0], points_reduced[:, 1], marker='o', markersize=3, color=colormap[1])
    plt.savefig('out/knees_trace_reduced.png', bbox_inches='tight', transparent=True)
    plt.savefig('out/knees_trace_reduced.pdf', bbox_inches='tight', transparent=True)
    plt.show()
    
    # Plot the original Knees
    knees = kneedle.knees(points_reduced, p=kneedle.PeakDetection.All)
    knees_original = rdp.mapping(knees, reduced, removed)
    plt.plot(points_reduced[:, 0], points_reduced[:, 1], marker='o', markersize=3, color=colormap[1])
    plt.plot(x[knees_original], y[knees_original], 's', markersize=5, color=colormap[2])
    plt.savefig('out/knees_original_knees.png', bbox_inches='tight', transparent=True)
    plt.savefig('out/knees_original_knees.pdf', bbox_inches='tight', transparent=True)
    plt.show()

    # Plot filtered and final knees
    knees = pp.filter_worst_knees(points_reduced, knees)
    knees = pp.filter_corner_knees(points_reduced, knees, t=0.33)
    knees = pp.filter_clusters(points_reduced, knees, clustering.average_linkage, 0.05, knee_ranking.ClusterRanking.left)
    knees = rdp.mapping(knees, reduced, removed)
    plt.plot(points_reduced[:, 0], points_reduced[:, 1], marker='o', markersize=3, color=colormap[1])
    plt.plot(x[knees_original], y[knees_original], 's', markersize=5, color=colormap[2])
    plt.plot(x[knees], y[knees], 'o', markersize=7, color=colormap[3])
    plt.savefig('out/knees_final_knees.png', bbox_inches='tight', transparent=True)
    plt.savefig('out/knees_final_knees.pdf', bbox_inches='tight', transparent=True)
    plt.show()
    
    # Plot original trace with final knees
    plt.plot(x, y, color= colormap[0])
    plt.plot(x[knees], y[knees], 'o', markersize=7, color=colormap[3])
    plt.savefig('out/knees_final_plot.png', bbox_inches='tight', transparent=True)
    plt.savefig('out/knees_final_plot.pdf', bbox_inches='tight', transparent=True)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the Multi-Knees using kneedle')
    parser.add_argument('-i', type=str, required=True, help='input file')
    args = parser.parse_args()
    main(args)