# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import os
import argparse
import numpy as np
import logging


from knee.kneedle import auto_knee
import knee.lmethod as lmethod
from knee.knee_ranking import *
from knee.postprocessing import filter_clustring
import matplotlib.pyplot as plt
from knee.rdp import rdp
import knee.clustering as clustering
from knee.multi_knee import multi_knee
import knee.max_perpendicular_distance as maxd

#import cProfile

from enum import Enum


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Method(Enum):
    kneedle = 'kneedle'
    lmethod = 'lmethod'
    maxd = 'maxd'

    def __str__(self):
        return self.value


class Clustering(Enum):
    single = 'single'
    complete = 'complete'
    average = 'average'

    def __str__(self):
        return self.value


# plot lines
def plot_lines(ax, x, y, title):
    ax.plot(x, y)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.text(.5,.9, title, horizontalalignment='center', transform=ax.transAxes)


# plot lines and knees (markers)
def plot_lines_knees(ax, x, y, knees, title):
    ax.plot(x, y)
    ax.plot(x[knees], y[knees], 'r+')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.text(.5,.9, title, horizontalalignment='center', transform=ax.transAxes)


def plot_kneedle(args, points, points_reduced, values, threshold):
    xpoints_reduced = points_reduced[:,0]
    ypoints_reduced = points_reduced[:,1]
    xdd = values['Dd'][:,0]
    ydd = values['Dd'][:,1]

    lines = [('differences', xdd, ydd), ('reduced', xpoints_reduced, ypoints_reduced)]

    for name, x, y in lines:
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2)
        logger.info('Plotting %s', name)
        plot_lines_knees(ax0, x, y, values['knees'], 'Knees Original')
        plot_lines_knees(ax1, x, y, values['knees_z'], 'Knees Z-Score')
        if len(values['knees_significant']) > 0:
            plot_lines_knees(ax2, x, y, values['knees_significant'], 'Knees Significant')
        plot_lines_knees(ax3, x, y, values['knees_iso'], 'Knees ISODATA')
    
        #plt.subplots_adjust(wspace=0, hspace=0)
        #plt.margins(0, 0)
        #filename = os.path.splitext(args.i)[0]+'.pdf'
        #plt.savefig(filename, transparent = True, bbox_inches = 'tight', pad_inches = 0, dpi = 300)
        #print('Plotting...')
        plt.show()
    
    #xpoints = np.transpose(points)[0]
    #ypoints = np.transpose(points)[1]
    #plot_lines(ax0, xpoints, ypoints, 'Original')

    
    #plot_lines(ax1, xpoints_reduced, ypoints_reduced, 'Reduced')

    #xds = np.transpose(values['Ds'])[0]
    #yds = np.transpose(values['Ds'])[1]
    #plot_lines(ax2, xds, yds, 'Smooth')

    
    #plot_lines(ax3, xdd, ydd, 'Differences')
    

    '''if 'zscores' in values:
        x_zscore = np.transpose(values['zscores'])[0]
        y_zscore = np.transpose(values['zscores'])[1]
        ax4.plot(x_zscore, y_zscore)
        x_zscore_left = np.transpose(values['zscores_left'])[0]
        y_zscore_left = np.transpose(values['zscores_left'])[1]
        ax4.plot(x_zscore_left, y_zscore_left, color='tab:orange')
        x_zscore_right = np.transpose(values['zscores_right'])[0]
        y_zscore_right = np.transpose(values['zscores_right'])[1]
        ax4.plot(x_zscore_right, y_zscore_right, color='tab:purple')
        ax4.set_yticklabels([])
        ax4.set_xticklabels([])
        ax4.axhline(y=threshold, color='r', linestyle='-')
        ax4.text(.5,.9,'Z-score',horizontalalignment='center', transform=ax4.transAxes)

        ax4_1 = ax4.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:green'
        ax4_1.plot(xdd, ydd, color=color)
        #ax4_1.tick_params(axis='y', labelcolor=color)'''
    

def ranking_to_color(ranking):
    color = (ranking, 0.5*(1.0-ranking), 1.0-ranking)
    return color


def get_dimention(lentgh: int):
    nrows = lentgh // 2
    ncols = lentgh - nrows
    return (nrows, ncols)


def plot_ranking_single(points, knees):
    fig, ax = plt.subplots()
    xpoints = points[:,0]
    ypoints = points[:,1]
    plot_lines(ax, xpoints, ypoints, 'knees')
    rankings_relative = slope_ranking(points, knees)
    for i in range(0, len(knees)):
        idx = knees[i]
        ax.axvline(xpoints[idx], color=ranking_to_color(rankings_relative[i]))
    fig.tight_layout()
    plt.show()


def plot_ranking(args, points, knees):
    xpoints = points[:,0]
    ypoints = points[:,1]

    if type(knees) != dict:
        plot_ranking_single(points, knees)
        return

    if args.m is Method.kneedle:
        keys = ['knees_z', 'knees', 'knees_significant', 'knees_iso']
    else:
        keys = ['knees']

    nrows = ncols = 1
    if len(keys) > 1:
        nrows, ncols = get_dimention(len(keys))
    fig, axs = plt.subplots(nrows, ncols)

    for j in range(0,  len(keys)):
        k = keys[j]
        
        if len(keys) == 1:
            plot_lines(axs, xpoints, ypoints, k)
        else:
            r = j % ncols
            c = j // ncols
            plot_lines(axs[r][c], xpoints, ypoints, k)
        rankings_relative = slope_ranking(points, knees[k])
        for i in range(0, len(knees[k])):
            idx = knees[k][i]
            if len(keys) == 1:
                axs.axvline(xpoints[idx], color=ranking_to_color(rankings_relative[i]))
            else:
                axs[r][c].axvline(xpoints[idx], color=ranking_to_color(rankings_relative[i]))

    filename = os.path.splitext(args.i)[0]+'_ranking.pdf'
    plt.savefig(filename, transparent = True, bbox_inches = 'tight', pad_inches = 0, dpi = 300)
    fig.tight_layout()
    plt.show()


def plot_lmethod(args, points, points_reduced, values):
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2)

    xpoints = points[:,0]
    ypoints = points[:,1]
    ax0.plot(xpoints, ypoints)
    ax0.set_yticklabels([])
    ax0.set_xticklabels([])
    ax0.text(.5,.9,'Original',horizontalalignment='center', transform=ax0.transAxes)

    xpoints_reduced = points_reduced[:,0]
    ypoints_reduced = points_reduced[:,1]
    ax1.plot(xpoints_reduced, ypoints_reduced)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.text(.5,.9,'Reduced',horizontalalignment='center', transform=ax1.transAxes)

    # compute lines
    xpoints = values['Dn'][:,0]
    ypoints = values['Dn'][:,1]

    knee_list = []
    for knee in values['knees']:
        #logger.info(knee)
        # knee index, left and right limit
        knee_idx = knee['knee']
        left = knee['left']
        right = knee['right'] - 1

        # compute left lines
        #left_coefficients = knee['coef_left']
        #poly = np.poly1d(left_coefficients)
        #x_left = np.linspace(xpoints[left], xpoints[knee_idx])
        #logger.info('X left = %s', x_left)
        #y_left = poly(x_left)
        #ax2.plot(x_left, y_left)
        ax2.plot([xpoints[left], xpoints[knee_idx]], [ypoints[left], ypoints[knee_idx]])

        # compute right lines
        #right_coefficients = knee['coef_right']
        #poly = np.poly1d(right_coefficients)
        #x_right = np.linspace(xpoints[knee_idx], xpoints[right])
        #logger.info('X right = %s', x_right)
        #y_right = poly(x_right)
        #ax2.plot(x_right, y_right)
        ax2.plot([xpoints[knee_idx], xpoints[right]], [ypoints[knee_idx], ypoints[right]])

        ax2.plot(xpoints[knee_idx], ypoints[knee_idx], 'ro')
        knee_list.append(knee_idx)

    #ax2.plot(xpoints, ypoints)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    #ax2.set_ylim(0, 1)
    ax2.text(.5, .9, 'Knees', horizontalalignment='center', transform=ax2.transAxes)

    #plot_lines_knees(ax3, xpoints_reduced, ypoints_reduced, knee_list, 'Knees')
    plot_lines_knees(ax3, xpoints, ypoints, knee_list, 'Knees')
    
    #plt.subplots_adjust(wspace=0, hspace=0)
    #plt.margins(0, 0)
    filename = os.path.splitext(args.i)[0]+'.pdf'
    plt.savefig(filename, transparent = True, bbox_inches = 'tight', pad_inches = 0, dpi = 300)
    logger.info('Plotting...')
    plt.show()


def plot_knees(points, knees):
    fig, ax = plt.subplots()

    x = points[:, 0]
    y = points[:, 1]
    plot_lines_knees(ax, x, y, knees, 'Knees')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def plot_points_removed(points, points_removed):
    fig, ax1 = plt.subplots()
    
    #plot original curve

    color = 'tab:blue'
    #ax1.set_xlabel('time (s)')
    #ax1.set_ylabel('exp', color=color)
    xpoints = points[:,0]
    ypoints = points[:,1]
    ax1.plot(xpoints, ypoints, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # plot removed points
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    xpoints = points_removed[:,0]
    ypoints = points_removed[:,1]
    #ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
    ax2.plot(xpoints, ypoints, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def postprocessing(points, knees, args):
    if type(knees) == dict:
        return postprocessing_dict(points, knees, args)
    else:
        return postprocessing_list(points, knees, args)


def postprocessing_dict(points, knees, args):
    logger.info('Post Processing')
    rv = knees.copy()
    
    if args.m is Method.kneedle:
        keys = ['knees', 'knees_z', 'knees_significant', 'knees_iso']
    else:
        keys = ['knees']

    for k in keys:
        logger.info('Knees: %s', k)
        current_knees = knees[k]
        logger.info('Initial Knees: %s', len(current_knees))
        #current_knees = filter_corner_point(points, current_knees, args.c)
        #logger.info('Corner Point Knees: %s', len(current_knees))
        if args.c is Clustering.single:
            current_knees = filter_clustring(points, current_knees, clustering.single_linkage, args.ct)
        elif args.c is Clustering.complete:
            current_knees = filter_clustring(points, current_knees, clustering.complete_linkage, args.ct)
        else:
            current_knees = filter_clustring(points, current_knees, clustering.average_linkage, args.ct)
        logger.info('Clustering Knees: %s', len(current_knees))
        rv[k] = np.array(current_knees)

    return rv


def postprocessing_list(points, knees, args):
    logger.info('Post Processing')
    rv = knees.copy()
    
    logger.info('Knees: %s', knees)
    logger.info('Initial #Knees: %s', len(knees))
    #current_knees = filter_corner_point(points, current_knees, args.c)
    #logger.info('Corner Point Knees: %s', len(current_knees))
    if args.c is Clustering.single:
        current_knees = filter_clustring(points, knees, clustering.single_linkage, args.ct)
    elif args.c is Clustering.complete:
        current_knees = filter_clustring(points, knees, clustering.complete_linkage, args.ct)
    else:
        current_knees = filter_clustring(points, knees, clustering.average_linkage, args.ct)
    logger.info('Clustering Knees: %s', len(current_knees))
    
    return current_knees



def main(args):
    points = np.genfromtxt(args.i, delimiter=',')

    #pr = cProfile.Profile()
    #pr.enable()
    points_reduced, points_removed = rdp(points, args.r)
    #pr.disable()
    #pr.print_stats()

    #double space_saving = MathUtils.round((1.0-(points_rdp.size()/(double)points.size()))*100.0, 2);
    space_saving = round((1.0-(len(points_reduced)/len(points)))*100.0, 2)
    logger.info('Number of data points after RDP: %s(%s %%)', len(points_reduced), space_saving)
    plot_points_removed(points, points_removed)

    #print(values)
    knees = None
    if args.m is Method.kneedle:
        knees = auto_knee(points_reduced, sensitivity=args.t, debug=True)
        plot_kneedle(args, points, points_reduced, knees, args.t)
    elif args.m is Method.lmethod:
        knees = lmethod.multiknee(points_reduced, args.lr, debug=True)
        plot_lmethod(args, points, points_reduced, knees)
        # convert knees to other format
        current_knees = knees['knees']
        other_format_knees = []
        for knee in current_knees:
            knee_idx = knee['knee']
            other_format_knees.append(knee_idx)
        knees['knees'] = np.array(other_format_knees)
    elif args.m is Method.maxd:
        knees = multi_knee(maxd.get_knee, points_reduced)
        plot_knees(points_reduced, knees)

    
    # plot rankings
    plot_ranking(args, points_reduced, knees)

    # post-processing
    filtered_knees = postprocessing(points_reduced, knees, args)
    #logger.info(filtered_knees)

    # plot rankings
    plot_ranking(args, points_reduced, filtered_knees)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi Knee testing app')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-r', type=float, help='RDP R2', default=0.95)
    parser.add_argument('--lr', type=float, help='L-Method R2', default=0.90)
    parser.add_argument('-t', type=float, help='Sensitivity', default=1.0)
    #parser.add_argument('-r', type=bool, help='Ranking relative', default=True)
    parser.add_argument('-m', type=Method, choices=list(Method), default='kneedle')
    parser.add_argument('-c', type=Clustering, choices=list(Clustering), default='single')
    parser.add_argument('--ct', type=float, help='clustering threshold', default=0.01)
    #parser.add_argument('-o', type=str, help='output file')
    args = parser.parse_args()
    
    main(args)