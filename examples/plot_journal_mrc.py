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
import numpy as np
import matplotlib.pyplot as plt


import kneeliverse.rdp as rdp
import kneeliverse.kneedle as kneedle
import kneeliverse.postprocessing as pp
import kneeliverse.clustering as clustering
import kneeliverse.knee_ranking as knee_ranking


#plt.style.use('seaborn-v0_8-paper')
plt.style.use('tableau-colorblind10')
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.figsize'] = (8, 4)
plt.rcParams['lines.linewidth'] = 2


logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def draw_brace(ax, xspan, yy, text, colormap):
    """Draws an annotated brace on the axes."""
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
    beta = 300./xax_span # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                    + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = yy + (.05*y - .01)*yspan # adjust vertical position

    ax.autoscale(False)
    ax.plot(x, -y, color='black', lw=2, clip_on=False)

    ax.text((xmax+xmin)/2., -yy-.2*yspan, text, ha='center', va='bottom', color=colormap[3])


def main():
    # Color Blind adjusted colors and markers
    colormap=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', 
    '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']
    markers=['o', '*', '.', 'x', '+', 's', 'd', 'h', 'v']
    lines=['-', ':', '--', '-.']
    
    points = np.genfromtxt('traces/web0.csv', delimiter=',')

    # Plot original trace and reduced version
    x = points[:,0]/1000
    y = points[:,1]
    
    plt.plot(x, y, color= colormap[0])
    plt.legend(['MRC (ARC)'])

    plt.xlabel('Cache Size (GB)')
    plt.ylabel('Miss Ratio')
    
    # Points
    pa = (36.4, 0.5417)
    pb = (53.7, 0.4508)
    pc = (90.2, 0.3956)
    pd = (162.9, 0.3469)

    # Arrows
    plt.annotate('A', xy=pa, weight='bold', color=colormap[7], 
    xytext=(pa[0]+10, pa[1]+0.05), arrowprops=dict(arrowstyle='->', lw=1.5, color=colormap[6]))
    plt.annotate('B', xy=pb, weight='bold', color=colormap[7], 
    xytext=(pb[0]+10, pb[1]+0.05), arrowprops=dict(arrowstyle='->', lw=1.5, color=colormap[6]))
    plt.annotate('C', xy=pc, weight='bold', color=colormap[7], 
    xytext=(pc[0]+10, pc[1]+0.05), arrowprops=dict(arrowstyle='->', lw=1.5, color=colormap[6]))
    plt.annotate('D', xy=pd, weight='bold', color=colormap[7], 
    xytext=(pd[0]+10, pd[1]+0.05), arrowprops=dict(arrowstyle='->', lw=1.5, color=colormap[6]))

    # Bracket
    #ax = plt.gca()
    #draw_brace(ax, (1.29E4, 3.10E4), -0.762, 'Range of cache\nspace with gradual\nmiss-ratio improvement', colormap)

    plt.savefig('out/mrc.png', bbox_inches='tight', transparent=True)
    plt.savefig('out/mrc.pdf', bbox_inches='tight', transparent=True)
    plt.show()


if __name__ == '__main__':
    main()