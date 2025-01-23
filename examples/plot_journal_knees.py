#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '1.0'
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

import kneeliverse.rdp as rdp
import kneeliverse.kneedle as kneedle
import kneeliverse.postprocessing as pp
import kneeliverse.clustering as clustering
import kneeliverse.knee_ranking as knee_ranking


#plt.style.use('seaborn-v0_8-paper')
plt.style.use('tableau-colorblind10')
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['lines.linewidth'] = 2


logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

def main(args):
    # Color Blind adjusted colors and markers
    colormap=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', 
    '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']
    markers=['o', '*', '.', 'x', '+', 's', 'd', 'h', 'v']
    lines=['-', ':', '--', '-.']

    points = np.genfromtxt(args.i, delimiter=',')
    logger.info(f'Loading {args.i} file ({len(points)})...')
    reduced, removed = rdp.mp_grdp(points, t=0.00075, min_points=20)
    space_saving = round((1.0-(len(reduced)/len(points)))*100.0, 2)
    logger.info('Number of data points after RDP: %s(%s %%)', len(reduced), space_saving)

    # Plot original trace and reduced version
    x = points[:,0]
    y = points[:,1]
    plt.plot(x, y, color= colormap[0])
    points_reduced = points[reduced]

    # save points to CSV
    np.savetxt(f'traces/web0_reduced.csv', points_reduced, delimiter=",")

    plt.plot(points_reduced[:, 0], points_reduced[:, 1], linestyle=lines[2], marker='o', markersize=3, color=colormap[1])
    plt.savefig('out/knees_trace_reduced.png', bbox_inches='tight', transparent=True)
    plt.savefig('out/knees_trace_reduced.pdf', bbox_inches='tight', transparent=True)
    plt.show()
    
    # Plot the original Knees
    knees = kneedle.knees(points_reduced, p=kneedle.PeakDetection.All)
    knees_original = rdp.mapping(knees, reduced, removed)
    plt.plot(points_reduced[:, 0], points_reduced[:, 1], linestyle=lines[2], marker='o', markersize=3, color=colormap[1])
    plt.plot(x[knees_original], y[knees_original], 's', markersize=5, color=colormap[2])
    plt.savefig('out/knees_original_knees.png', bbox_inches='tight', transparent=True)
    plt.savefig('out/knees_original_knees.pdf', bbox_inches='tight', transparent=True)
    plt.show()

    # Plot filtered and final knees
    previous_knees_len = len(knees)
    previous_knees = set(knees)
    
    knees = pp.filter_worst_knees(points_reduced, knees)
    if len(knees) < previous_knees_len:
        diff = previous_knees.difference(set(knees))
        logger.debug(f'Filter worst knees removed {previous_knees_len-len(knees)} ({diff})')
        
        previous_knees = set(knees)
        previous_knees_len = len(knees)
        
        knees_worst = rdp.mapping(knees, reduced, removed)
        plt.plot(points_reduced[:, 0], points_reduced[:, 1], linestyle=lines[2], marker='o', markersize=3, color=colormap[1])
        plt.plot(x[knees_original], y[knees_original], 's', markersize=5, color=colormap[2])
        plt.plot(x[knees_worst], y[knees_worst], 'o', markersize=7, color=colormap[7])
        plt.show()

    knees = pp.filter_corner_knees(points_reduced, knees, t=0.33)
    if len(knees) < previous_knees_len:
        diff = previous_knees.difference(set(knees))
        logger.debug(f'Filter corner knees removed {previous_knees_len-len(knees)} ({diff})')
        previous_knees = set(knees)
        previous_knees_len = len(knees)
        knees_corner = rdp.mapping(knees, reduced, removed)
        plt.plot(points_reduced[:, 0], points_reduced[:, 1], linestyle=lines[2], marker='o', markersize=3, color=colormap[1])
        plt.plot(x[knees_worst], y[knees_worst], 's', markersize=5, color=colormap[2])
        plt.plot(x[knees_corner], y[knees_corner], 'o', markersize=7, color=colormap[7])
        plt.show()
    
    knees = pp.filter_clusters(points_reduced, knees, clustering.average_linkage, 0.05, knee_ranking.ClusterRanking.left)
    if len(knees) < previous_knees_len:
        diff = previous_knees.difference(set(knees))
        logger.debug(f'Filter cluster removed {previous_knees_len-len(knees)} ({diff})')
        previous_knees = set(knees)
        previous_knees_len = len(knees)
        knees_cluster = rdp.mapping(knees, reduced, removed)
        plt.plot(points_reduced[:, 0], points_reduced[:, 1], linestyle=lines[2], marker='o', markersize=3, color=colormap[1])
        plt.plot(x[knees_corner], y[knees_corner], 's', markersize=5, color=colormap[2])
        plt.plot(x[knees_cluster], y[knees_cluster], 'o', markersize=7, color=colormap[7])
        plt.show()
    
    knees = rdp.mapping(knees, reduced, removed)
    plt.plot(points_reduced[:, 0], points_reduced[:, 1], linestyle=lines[2], marker='o', markersize=3, color=colormap[1])
    plt.plot(x[knees_original], y[knees_original], 's', markersize=5, color=colormap[2])
    plt.plot(x[knees], y[knees], 'o', markersize=7, color=colormap[7])
    plt.savefig('out/knees_final_knees.png', bbox_inches='tight', transparent=True)
    plt.savefig('out/knees_final_knees.pdf', bbox_inches='tight', transparent=True)
    plt.show()
    
    # Plot original trace with final knees
    plt.plot(x, y, color= colormap[0])
    plt.plot(x[knees], y[knees], 'o', markersize=7, color=colormap[7])
    plt.savefig('out/knees_final_plot.png', bbox_inches='tight', transparent=True)
    plt.savefig('out/knees_final_plot.pdf', bbox_inches='tight', transparent=True)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the Multi-Knees using kneedle')
    parser.add_argument('-i', type=str, required=True, help='input file')
    args = parser.parse_args()
    main(args)