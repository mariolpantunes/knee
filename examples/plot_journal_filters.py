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


import logging
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from matplotlib.patches import Ellipse


#plt.style.use('seaborn-v0_8-paper')
plt.style.use('tableau-colorblind10')
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['lines.linewidth'] = 2


def main():
    # Color Blind adjusted colors and markers
    colormap=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', 
    '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']
    markers=['o', '*', '.', 'x', '+', 's', 'd', 'h', 'v']
    lines=['-', ':', '--', '-.']

    x1 = [0, 0.5, 3,   4, 4.25,   5]
    y1 = [5,   1, 4, 4.25,   2, 0.5]

    # Plot 1
    fg, ax = plt.subplots()
    ax.plot(x1, y1, color=colormap[0], marker=markers[0], markersize=3)

    ax.axhline(y = 1, color=colormap[1], linestyle=lines[2])
    ax.axhline(y = 2, color=colormap[2], linestyle=lines[2])
    ax.annotate('K0', (0.70, 1.1), color=colormap[1])
    ax.annotate('K1', (4.30, 2.1), color=colormap[2])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.savefig('out/knee_post_processing_00.png', bbox_inches='tight', transparent=True)
    plt.savefig('out/knee_post_processing_00.pdf', bbox_inches='tight', transparent=True)
    plt.show()

    x2 = [0,2,2.1,2.2,3,5]
    y2 = [5,4.5,4,1.5,.25,0]
    
    # Plot 2
    fg, ax = plt.subplots()
    ax.plot(x2, y2, color=colormap[0], marker=markers[0], markersize=3)
    ax.add_patch(Rectangle((0, 4), 2, .5, fill=False, hatch='/', color=colormap[1], linewidth=2))
    ax.add_patch(Rectangle((0, 4), 2.1, 1, fill=False, hatch='\\', color=colormap[2], linewidth=2))

    ax.plot(0, 5, 'o', ms=7, color=colormap[2])
    ax.annotate('P0', (0.1, 5.1), color=colormap[2])
    ax.plot(2, 4.5, 'X', ms=7, color=colormap[1])
    ax.annotate('C', (2.2, 4.6), color=colormap[1])
    ax.plot(2.1, 4, 'o', ms=7, color=colormap[2])
    ax.annotate('P1', (2.2, 4.1), color=colormap[2])
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.savefig('out/knee_post_processing_01.png', bbox_inches='tight', transparent=True)
    plt.savefig('out/knee_post_processing_01.pdf', bbox_inches='tight', transparent=True)
    plt.show()

    # Plot 3
    fg, ax = plt.subplots()
    ax.plot(x2, y2, color=colormap[0], marker=markers[0], markersize=3)

    ax.add_patch(Ellipse((2.2, 1.5), 4, 1.5, color=colormap[6], angle=-75, alpha=0.3))

    ax.plot(2.1, 3, 'x', mew=5, ms=10, color=colormap[1])
    ax.plot(2.15, 2.25, 'x', mew=5, ms=10, color=colormap[1])

    ax.plot(2.2, 1.5, 'x', mew=5, ms=10, color=colormap[2])

    ax.plot(2.5, 1.0315, 'x', mew=5, ms=10, color=colormap[1])
    ax.plot(2.8, 0.5626, 'x', mew=5, ms=10, color=colormap[1])

    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    plt.savefig('out/knee_post_processing_02.png', bbox_inches='tight', transparent=True)
    plt.savefig('out/knee_post_processing_02.pdf', bbox_inches='tight', transparent=True)
    plt.show()


if __name__ == '__main__':
    main()