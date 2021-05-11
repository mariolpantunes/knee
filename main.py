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
import knee.max_perpendicular_distance as maxpd
import knee.knee_ranking as ranking 
from plot import get_dimention, plot_lines_knees_ranking, plot_lines_knees


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_knees_ranking(points, knees, names):
    x = points[:,0]
    y = points[:,1]

    nrows = ncols = 1
    
    if len(knees) > 1:
        nrows, ncols = get_dimention(len(knees))
    
    fig, axs = plt.subplots(nrows, ncols)

    for j in range(len(knees)):
        rankings = ranking.slope_ranking(points, knees[j])
        if len(knees) == 1:
            plot_lines_knees_ranking(axs, x, y, knees[j], rankings, names[j])
        elif nrows == 1:
            plot_lines_knees_ranking(axs[j], x, y, knees[j], rankings, names[j])
        else:
            c = j % ncols
            r = j // ncols
            plot_lines_knees_ranking(axs[r, c], x, y, knees[j], rankings, names[j])
        

    #filename = os.path.splitext(args.i)[0]+'_ranking.pdf'
    #plt.savefig(filename, transparent = True, bbox_inches = 'tight', pad_inches = 0, dpi = 300)
    #fig.tight_layout()
    plt.show()


def plot_knees(points, knees, names):
    x = points[:,0]
    y = points[:,1]

    nrows = ncols = 1
    
    if len(knees) > 1:
        nrows, ncols = get_dimention(len(knees))
    
    fig, axs = plt.subplots(nrows, ncols)

    for j in range(len(knees)):
        
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

    names = ['kneedle', 'kneedke(Rec)', 'l-method', 'dfdt', 'menger', 'curvature', 'maxpd', 'fusion', 'rdp'] 
    methods = [kneedle.auto_knee, kneedle.multi_knee, lmethod.multi_knee, dfdt.multi_knee, menger.multi_knee, curvature.multi_knee, maxpd.multi_knee]
    knees = []

    # Elbow methods
    for m in methods:
        knees.append(m(points_reduced))
    
    # Fusion
    fusion_knees = np.unique(np.concatenate(knees, 0))
    knees.append(fusion_knees)
    
    # RDP
    knees.append(np.arange(1, len(points_reduced)))

    plot_knees(points_reduced, knees, names)

    filtered_knees = []
    for k in knees:
        t_k = pp.filter_worst_knees(points_reduced, k)
        filtered_knees.append(pp.filter_clustring(points_reduced, t_k, clustering.average_linkage, 0.02))

    plot_knees_ranking(points_reduced, filtered_knees, names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi Knee evaluation app')
    parser.add_argument('-i', type=str, required=True, help='input file')
    args = parser.parse_args()
    
    main(args)