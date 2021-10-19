#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import argparse
import numpy as np
import logging
import matplotlib.pyplot as plt


def main(args):
    # adjust matplotlib parameters
    #plt.rcParams.update(plt.rcParamsDefault)
    #plt.rcParams['text.usetex'] = True
    #plt.rcParams['font.size'] = 18
    #plt.rcParams['font.family'] = "serif"

    # Create main container
    fig = plt.figure(figsize=(4, 1.75))
    #fig = plt.figure()

    #ax = plt.gca() #you first need to get the axis handle
    #ax.set_aspect(1.75/4)

    points = np.genfromtxt(args.i, delimiter=',')
    x = points[:, 0]
    y = points[:, 1]

    x = (x*4096)/1073741824

    # Create zoom-in plot
    plt.plot(x, y)
    #plt.xlim(400, 500)
    #plt.ylim(350, 400)
    plt.xlabel('Cache Size (GB)', labelpad = 15)
    plt.ylabel('Miss Ratio', labelpad = 15)
    plt.ylim(0,1)
    

    # Create zoom-out plot
    ax_new = fig.add_axes([0.65, 0.65, 0.2, 0.2]) # the position of zoom-out plot compare to the ratio of zoom-in plot 
    t = 0.1
    idx = int(len(points)*(1.0-t))
    print(f'{len(points)} -> {idx}')
    
    ax_new.axes.yaxis.set_ticklabels([])

    plt.plot(x[idx:], y[idx:])
    #plt.xlim(0, 850)
    #plt.ylim(0, 1)
    #plt.xlabel('Cache Size (GB)', labelpad = 15)
    #plt.ylabel('Miss Ratio', labelpad = 15)
    

    # Save figure with nice margin
    plt.savefig('/home/mantunes/Desktop/resolution.pdf', dpi = 300, bbox_inches = 'tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RDP test application')
    parser.add_argument('-i', type=str, required=True, help='input file')
    args = parser.parse_args()
    
    main(args)