#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import os
import argparse
import numpy as np
import logging


import knee.rdp as rdp
import knee.kneedle as kneedle
import knee.lmethod as lmethod
import knee.dfdt as dfdt
import knee.menger as menger
import knee.curvature as curvature
import knee.postprocessing as pp
import matplotlib.pyplot as plt
import knee.clustering as clustering
import knee.pointSelector as ps
import knee.knee_ranking as ranking 
from plot import get_dimention, plot_lines_knees_ranking, plot_lines_knees


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def plot_knees_ranking(points, knees, names, rankings):
    x = points[:,0]
    y = points[:,1]

    nrows = ncols = 1
    
    if len(knees) > 1:
        nrows, ncols = get_dimention(len(knees))
    
    _, axs = plt.subplots(nrows, ncols)

    for j in range(len(knees)):
        if len(knees) == 1:
            plot_lines_knees_ranking(axs, x, y, knees[j], rankings[j], names[j])
        elif nrows == 1:
            plot_lines_knees_ranking(axs[j], x, y, knees[j], rankings[j], names[j])
        else:
            c = j % ncols
            r = j // ncols
            plot_lines_knees_ranking(axs[r, c], x, y, knees[j], rankings[j], names[j])
        

    #filename = os.path.splitext(args.i)[0]+'_ranking.pdf'
    #plt.savefig(filename, transparent = True, bbox_inches = 'tight', pad_inches = 0, dpi = 300)
    #fig.tight_layout()
    plt.show()


def plot_knees(points, knees, names):
    x = points[:,0]
    y = points[:,1]

    nrows = ncols = 1
    
    if len(names) > 1:
        nrows, ncols = get_dimention(len(names))
    
    _, axs = plt.subplots(nrows, ncols)

    for j in range(len(names)):
        
        if len(knees) == 1:
            plot_lines_knees(axs, x, y, knees[j], names[j])
        elif nrows == 1:
            plot_lines_knees(axs[j], x, y, knees[j], names[j])
        else:
            c = j % ncols
            r = j // ncols
            plot_lines_knees(axs[r, c], x, y, knees[j], names[j])
        

    #filename = os.path.splitext(args.i)[0]+'_ranking.pdf'
    #plt.savefig(filename, transparent = True, bbox_inches = 'tight', pad_inches = 0, dpi = 300)
    #fig.tight_layout()
    plt.show()


def main(args):
    points = np.genfromtxt(args.i, delimiter=',')
    points_reduced, points_removed = rdp.rdp(points, 0.95)

    space_saving = round((1.0-(len(points_reduced)/len(points)))*100.0, 2)
    logger.info('Number of data points after RDP: %s(%s %%)', len(points_reduced), space_saving)

    names = ['kneedle', 'kneedke(Rec)', 'l-method', 'dfdt', 'menger', 'curvature', 'Tyler (RDP)', 'Tyler', 'rdp'] 
    methods = [kneedle.auto_knee, kneedle.multi_knee, lmethod.multi_knee, dfdt.multi_knee, menger.multi_knee, curvature.multi_knee, ps.get_knees_points]
    knees = []
    knees_raw = []

    # Elbow methods
    for m in methods:
        tmp = m(points_reduced)
        knees.append(tmp)
        raw_indexes = rdp.mapping(tmp, points_reduced, points_removed)
        knees_raw.append(raw_indexes)
    
    # Tyler
    candidates = ps.get_knees_points(points)
    knees.append(candidates)
    knees_raw.append(candidates)
    
    # Fusion
    #fusion_knees = np.unique(np.concatenate(knees, 0))
    #knees.append(fusion_knees)
    
    # RDP
    candidates = np.arange(1, len(points_reduced))
    knees.append(candidates)
    raw_indexes = rdp.mapping(candidates, points_reduced, points_removed)
    knees_raw.append(raw_indexes)

    plot_knees(points, knees_raw, names)

    # Cluster and select points
    filtered_knees_raw = []
    rankings = []
    for k, n in zip(knees, names):
        if n == 'Tyler':
            filtered_knees_raw.append(k)
            ranks = np.full(len(k), 1.0)
            #rankings.append(ranking.slope_ranking(points, k))
            rankings.append(ranks)
        else:
            t_k = pp.filter_worst_knees(points_reduced, k)
            filtered_knees = pp.filter_clustring(points_reduced, t_k, clustering.single_linkage, 0.01)
            rankings.append(ranking.slope_ranking(points_reduced, filtered_knees))
            raw_indexes = rdp.mapping(filtered_knees, points_reduced, points_removed)
            filtered_knees_raw.append(raw_indexes)
    
    plot_knees_ranking(points, filtered_knees_raw, names, rankings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi Knee evaluation app')
    parser.add_argument('-i', type=str, required=True, help='input file')
    args = parser.parse_args()
    
    main(args)