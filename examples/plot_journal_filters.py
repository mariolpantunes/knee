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


import logging
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import transforms
from matplotlib.patches import Rectangle
from matplotlib.patches import Ellipse


#plt.style.use('seaborn-v0_8-paper')
plt.style.use('tableau-colorblind10')
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['lines.linewidth'] = 2


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ## Using a special case to obtain the eigenvalues of this
    ## two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    ## Calculating the standard deviation of x from
    ## the squareroot of the variance and multiplying
    ## with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    ## calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def main():
    # Color Blind adjusted colors and markers
    colormap=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', 
    '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']
    markers=['o', '*', '.', 'x', '+', 's', 'd', 'h', 'v']
    lines=['-', ':', '--', '-.']

    points = np.genfromtxt(f'traces/web0_reduced.csv', delimiter=',')

    worst_knee = 20
    x1 = points[worst_knee-5:worst_knee+3,0]
    y1 = points[worst_knee-5:worst_knee+3,1]

    # Plot 1
    fg, ax = plt.subplots()
    ax.plot(x1, y1, color=colormap[0], marker=markers[0], markersize=3)

    ax.axhline(y = 0.5909, color=colormap[1], linestyle=lines[2])
    ax.axhline(y = 0.6127, color=colormap[2], linestyle=lines[2])
    ax.annotate('K0', (2.474E4, 0.5922), color=colormap[1])
    ax.annotate('K1', (2.893E4, 0.6140), color=colormap[2])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.savefig('out/knee_post_processing_00.png', bbox_inches='tight', transparent=True)
    plt.savefig('out/knee_post_processing_00.pdf', bbox_inches='tight', transparent=True)
    plt.show()

    corner_knee = 45
    x2 = points[corner_knee-3:corner_knee+3,0]
    y2 = points[corner_knee-3:corner_knee+3,1]
    cop = (8.922E4, 0.4048)
    prp = (7.32E4, 0.4057)
    nep = (9.016E4, 0.3955)
    arp = (prp[0], nep[1])
    w1 = cop[0] - arp[0]
    h1 = cop[1] - arp[1]
    w2 = nep[0] - arp[0]
    h2 = prp[1] - arp[1]
    
    # Plot 2
    fg, ax = plt.subplots()
    ax.plot(x2, y2, color=colormap[0], marker=markers[0], markersize=3)
    ax.add_patch(Rectangle(arp, w1, h1, fill=False, hatch='/', color=colormap[1], linewidth=2))
    ax.add_patch(Rectangle(arp, w2, h2, fill=False, hatch='\\', color=colormap[2], linewidth=2))
    
    ax.plot(prp[0], prp[1], 'o', ms=7, color=colormap[2])
    ax.annotate('P0', (prp[0], prp[1]+0.0007), color=colormap[2])
    ax.plot(cop[0], cop[1], 'X', ms=7, color=colormap[1])
    ax.annotate('C', (cop[0]+2000, cop[1]), color=colormap[1])
    ax.plot(nep[0], nep[1], 'o', ms=7, color=colormap[2])
    ax.annotate('P1', (cop[0]+2000, nep[1]), color=colormap[2])
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.savefig('out/knee_post_processing_01.png', bbox_inches='tight', transparent=True)
    plt.savefig('out/knee_post_processing_01.pdf', bbox_inches='tight', transparent=True)
    plt.show()

    # Plot 3
    cluster_knee = 37
    x3 = points[cluster_knee:cluster_knee+6,0]
    y3 = points[cluster_knee:cluster_knee+6,1]

    fg, ax = plt.subplots()
    ax.plot(x3, y3, color=colormap[0], marker=markers[0], markersize=3)

    ecx = np.array([5.374e+04, 5.384e+04, 5.55e+04])
    ecy = np.array([4.270992757584038957e-01, 4.263545367586771828e-01, 4.220090871822902434e-01])

    confidence_ellipse(ecx, ecy, ax, n_std=2.5, 
    **{'color':colormap[6], 'angle':0, 'alpha':0.3})

    ax.plot(5.374e+04,4.270992757584038957e-01, 'x', mew=5, ms=10, color=colormap[1])
    ax.plot(5.384e+04,4.263545367586771828e-01, 'x', mew=5, ms=10, color=colormap[2])
    ax.plot(5.55e+04,4.220090871822902434e-01, 'x', mew=5, ms=10, color=colormap[1])

    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    plt.savefig('out/knee_post_processing_02.png', bbox_inches='tight', transparent=True)
    plt.savefig('out/knee_post_processing_02.pdf', bbox_inches='tight', transparent=True)
    plt.show()


if __name__ == '__main__':
    main()