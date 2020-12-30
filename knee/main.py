# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import os
import csv
import argparse
import numpy as np

from rdp import rdp
from kneedle import auto_knee
from lmethod import knee
from knee_ranking import *
import matplotlib.pyplot as plt

#import cProfile

from enum import Enum


class Method(Enum):
    kneedle = 'kneedle'
    lmethod = 'lmethod'

    def __str__(self):
        return self.value


def plot_kneedle(args, points, points_reduced, values, threshold):
    fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(3,2)
    
    xpoints = np.transpose(points)[0]
    ypoints = np.transpose(points)[1]
    ax0.plot(xpoints, ypoints)
    ax0.set_yticklabels([])
    ax0.set_xticklabels([])
    ax0.text(.5,.9,'Original',horizontalalignment='center', transform=ax0.transAxes)

    xpoints_reduced = np.transpose(points_reduced)[0]
    ypoints_reduced = np.transpose(points_reduced)[1]
    ax1.plot(xpoints_reduced, ypoints_reduced)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.text(.5,.9,'Reduced',horizontalalignment='center', transform=ax1.transAxes)

    xds = np.transpose(values['Ds'])[0]
    yds = np.transpose(values['Ds'])[1]
    ax2.plot(xds, yds)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.text(.5,.9,'Smooth',horizontalalignment='center', transform=ax2.transAxes)

    xdd = np.transpose(values['Dd'])[0]
    ydd = np.transpose(values['Dd'])[1]
    ax3.plot(xdd, ydd)
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax3.text(.5,.9,'Differences',horizontalalignment='center', transform=ax3.transAxes)

    if 'zscores' in values:
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
        #ax4_1.tick_params(axis='y', labelcolor=color)

    xpoints_reduced = np.transpose(points_reduced)[0]
    ypoints_reduced = np.transpose(points_reduced)[1]
    ax5.plot(xpoints_reduced, ypoints_reduced)
    ax5.plot(xpoints_reduced[values['knees']], ypoints_reduced[values['knees']], 'r+')
    ax5.set_yticklabels([])
    ax5.set_xticklabels([])
    ax5.text(.5,.9,'Knees',horizontalalignment='center', transform=ax5.transAxes)

    #for i in values['knees']:
    #    ax5.axvline(xpoints_reduced[i], color='r')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(0, 0)
    filename = os.path.splitext(args.i)[0]+'.pdf'
    plt.savefig(filename, transparent = True, bbox_inches = 'tight', pad_inches = 0, dpi = 300)
    print('Plotting...')
    plt.show()


def ranking_to_color(ranking):
    color = (ranking, 0.5*(1.0-ranking), 1.0-ranking)
    return color


def plot_ranking(args, points, knees):
    rankings_k = curvature_ranking(points, knees)
    rankings_m = menger_curvature_ranking(points, knees)
    rankings_l = slope_ranking(points, knees)
    rankings_d = dfdt_ranking(points, knees)
    
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2)
    
    xpoints = np.transpose(points)[0]
    ypoints = np.transpose(points)[1]
    ax0.plot(xpoints, ypoints)
    ax0.set_yticklabels([])
    ax0.set_xticklabels([])
    ax0.text(.5, .9, 'Curvature', horizontalalignment='center', transform=ax0.transAxes)
    ax0.margins(0, 0)

    xpoints = np.transpose(points)[0]
    ypoints = np.transpose(points)[1]
    ax1.plot(xpoints, ypoints)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.text(.5, .9, 'Menger', horizontalalignment='center', transform=ax1.transAxes)
    ax1.margins(0, 0)

    xpoints = np.transpose(points)[0]
    ypoints = np.transpose(points)[1]
    ax2.plot(xpoints, ypoints)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.text(.5, .9, 'Slope', horizontalalignment='center', transform=ax2.transAxes)
    ax2.margins(0, 0)

    xpoints = np.transpose(points)[0]
    ypoints = np.transpose(points)[1]
    ax3.plot(xpoints, ypoints)
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax3.text(.5, .9, 'DFDT', horizontalalignment='center', transform=ax3.transAxes)
    ax3.margins(0, 0)
    
    for i in range(0, len(knees)):
        idx = knees[i]
        ax0.axvline(xpoints[idx], color=ranking_to_color(rankings_k[i]))
        ax1.axvline(xpoints[idx], color=ranking_to_color(rankings_m[i]))
        ax2.axvline(xpoints[idx], color=ranking_to_color(rankings_l[i]))
        ax3.axvline(xpoints[idx], color=ranking_to_color(rankings_d[i]))

    filename = os.path.splitext(args.i)[0]+'_ranking.pdf'
    plt.savefig(filename, transparent = True, bbox_inches = 'tight', pad_inches = 0, dpi = 300)
    print('Plotting...')
    fig.tight_layout()
    plt.show()


def plot_lmethod(args, points, points_reduced, values):
    pass


def plot_points_removed(points, points_removed):
    fig, ax1 = plt.subplots()
    
    #plot original curve

    color = 'tab:blue'
    #ax1.set_xlabel('time (s)')
    #ax1.set_ylabel('exp', color=color)
    xpoints = np.transpose(points)[0]
    ypoints = np.transpose(points)[1]
    ax1.plot(xpoints, ypoints, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # plot removed points
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    xpoints = np.transpose(points_removed)[0]
    ypoints = np.transpose(points_removed)[1]
    #ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
    ax2.plot(xpoints, ypoints, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def main(args):
    points = np.genfromtxt(args.i, delimiter=',')
    print(points)

    #pr = cProfile.Profile()
    #pr.enable()
    points_reduced, points_removed = rdp(points, args.r2)
    #pr.disable()
    #pr.print_stats()

    print(points)
    #double space_saving = MathUtils.round((1.0-(points_rdp.size()/(double)points.size()))*100.0, 2);
    space_saving = round((1.0-(len(points_reduced)/len(points)))*100.0, 2)
    print('Number of data points after RDP: {}({}%)'.format(len(points_reduced), space_saving))
    plot_points_removed(points, points_removed)


    #print(values)
    knees = None
    if args.m is Method.kneedle:
        knees = auto_knee(points_reduced, sensitivity=args.t, debug=True)
        plot_kneedle(args, points, points_reduced, knees, args.t)
    elif args.m is Method.lmethod:
        plot_lmethod(args, points, points_reduced, knee(points_reduced, debug=True))
    
    # plot rankings
    plot_ranking(args, points_reduced, knees['knees'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi Knee testing app')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('--r2', type=float, help='R2', default=0.95)
    parser.add_argument('-t', type=float, help='Sensitivity', default=1.0)
    parser.add_argument('-r', type=bool, help='Ranking relative', default=True)
    parser.add_argument('-m', type=Method, choices=list(Method), default='kneedle')
    #parser.add_argument('-o', type=str, help='output file')
    args = parser.parse_args()
    
    main(args)