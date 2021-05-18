# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
from knee.knee_ranking import rank


def get_dimention(lentgh: int):
    nrows = int(round(math.sqrt(lentgh)))
    ncols = int(math.ceil(lentgh / nrows))
    return (nrows, ncols)


def plot_lines_knees(ax, x, y, knees, title):
    ax.plot(x, y)
    ax.plot(x[knees], y[knees], 'r+')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.text(.5,.9, title, horizontalalignment='center', transform=ax.transAxes)


def ranking_to_color(ranking):
    color = (ranking, 0.5*(1.0-ranking), 1.0-ranking)
    return color


def plot_lines_knees_ranking(ax, x, y, knees, rankings, title):
    ax.plot(x, y)
    for i in range(0, len(knees)):
        idx = knees[i]
        ax.plot([x[idx]], [y[idx]], marker='o', markersize=3, color=ranking_to_color(rankings[i]))
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.text(.5,.9, title, horizontalalignment='center', transform=ax.transAxes)


def plot_lines(ax, x, y, title):
    ax.plot(x, y)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.text(.5,.9, title, horizontalalignment='center', transform=ax.transAxes)


def plot_ranking(plt, points, knees, ranking, title):
    fig, ax = plt.subplots()
    xpoints = points[:,0]
    ypoints = points[:,1]
    plot_lines(ax, xpoints, ypoints, title)
    #rankings_relative = ranking(points, knees)
    
    for i in range(0, len(knees)):
        idx = knees[i]
        ax.plot([xpoints[idx]], [ypoints[idx]], marker='o', markersize=3, color=ranking_to_color(ranking[i]))

    fig.tight_layout()
    #fig.suptitle(title)
