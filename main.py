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
#from knee.knee_ranking import *
#from knee.postprocessing import filter_clustring
import matplotlib.pyplot as plt

import knee.clustering as clustering


from plot import get_dimention, plot_lines_knees

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_knees(points, knees, names):
    x = points[:,0]
    y = points[:,1]

    nrows = ncols = 1
    
    if len(knees) > 1:
        nrows, ncols = get_dimention(len(knees))
    
    logger.info('%s x %s', nrows, ncols)
    
    fig, axs = plt.subplots(nrows, ncols)

    for j in range(len(knees)): 
        logger.info('%s', knees[j])
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

    names = ['kneedle', 'l-method']
    methods = [kneedle.auto_knee, lmethod.multiknee]
    knees = []

    for m in methods:
        knees.append(m(points_reduced))
    
    plot_knees(points_reduced, knees, names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi Knee evaluation app')
    parser.add_argument('-i', type=str, required=True, help='input file')
    #parser.add_argument('-r', type=float, help='RDP R2', default=0.95)
    #parser.add_argument('--lr', type=float, help='L-Method R2', default=0.90)
    #parser.add_argument('-t', type=float, help='Sensitivity', default=1.0)
    #parser.add_argument('-r', type=bool, help='Ranking relative', default=True)
    #parser.add_argument('-m', type=Method, choices=list(Method), default='kneedle')
    #parser.add_argument('-c', type=Clustering, help=list(Clustering), default='single')
    #parser.add_argument('--ct', type=float, help='clustering threshold', default=0.01)
    #parser.add_argument('-o', type=str, help='output file')
    args = parser.parse_args()
    
    main(args)