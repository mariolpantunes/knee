# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math


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
