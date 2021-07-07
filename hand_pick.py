#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import os
import csv
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Global variables
fig, ax = plt.subplots()
points = None
dataset = None


def onclick(event):
    global dataset
    global points

    # clear the plot
    ax.cla() 
    
    # redraw the points
    x = points[:,0]
    y = points[:,1]
    ax.plot(x, y)

    # get the x coordinate from the clicked point
    ex = event.xdata
    
    # add or remove the point from the dataset
    if event.button is MouseButton.LEFT:
        # find the closest point to the original points, using the X axis
        x_distances = np.abs(x - ex)
        closest_idx = np.argmin(x_distances)
        cx = x[closest_idx]
        cy = y[closest_idx]
        dataset.append([cx, cy])

        logger.info('Add a Point %s', [cx, cy])

    elif event.button is MouseButton.RIGHT:
        # find the closest point to the selected points, using the X axis
        if len(dataset) > 0:
            d = np.array(dataset)
            dx = d[:,0]
            x_distances = np.abs(dx - ex)
            closest_idx = np.argmin(x_distances)
            cx = dataset[closest_idx][0]
            cy = dataset[closest_idx][1]
            dataset.remove([cx, cy])

        logger.info('Remove a Point %s', [cx, cy])
    
    logger.info('Dataset = %s', dataset)

    # redraw the dataset
    for x,y in dataset:
        ax.plot(x, y, marker='o', markersize=3, color='red')

    fig.canvas.draw()


def main(args):
    # get the expected file from the input file
    dirname = os.path.dirname(args.i)
    filename = os.path.splitext(os.path.basename(args.i))[0]
    output = os.path.join(os.path.normpath(dirname), f'{filename}_expected.csv')

    # trying to load the dataset
    global dataset

    if os.path.exists(output):
        with open(output, 'r') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            dataset = list(reader)
    else:
        dataset = []
    logger.info('Loaded dataset: %s', dataset)

    global points
    points = np.genfromtxt(args.i, delimiter=',')

    x = points[:,0]
    y = points[:,1]
    ax.plot(x, y)

    for x,y in dataset:
        logger.info('(%s, %s)', x, y)
        ax.plot(x, y, marker='o', markersize=3, color='red')
    
    _ = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Store the dataset into a CSV
    with open(output, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(dataset) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hand Pick dataset generator')
    parser.add_argument('-i', type=str, required=True, help='input file')
    args = parser.parse_args()
    
    main(args)