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
import knee.clustering as clustering

from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class Clustering(Enum):
    single = 'single'
    complete = 'complete'
    average = 'average'

    def __str__(self):
        return self.value


def rank_cluster_points(points, clusters):
    max_cluster = clusters.max()
    filtered_points = []
    for i in range(0, max_cluster+1):
        current_cluster = points[clusters == i]

        if len(current_cluster) > 1:
            #select point with lower y value
            idx = np.argmin(current_cluster, axis=0)[1]
            best_point = current_cluster[idx]
        else:
            best_point = points[clusters == i][0]
        filtered_points.append(best_point)

    return np.array(filtered_points)


def main(args):
    # get the expected file from the input file
    dirname = os.path.dirname(args.i)
    filename = os.path.splitext(os.path.basename(args.i))[0]
    expected_file = os.path.join(os.path.normpath(dirname), f'{filename}_expected.csv')

    # trying to load the dataset
    dataset = []

    if os.path.exists(expected_file):
        with open(expected_file, 'r') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            dataset = list(reader)
    dataset = np.array(dataset)
        
    logger.info(f'Loaded dataset ({len(dataset)})')

    points = np.genfromtxt(args.i, delimiter=',')

    fig, axs = plt.subplots(2)
    #fig.suptitle('Vertically stacked subplots')
    #axs[0].plot(x, y)
    #axs[1].plot(x, -y)

    x = points[:,0]
    y = points[:,1]
    axs[0].plot(x, y)

    for x,y in dataset:
        
        axs[0].plot(x, y, marker='o', markersize=3, color='red')
    
    # Cluster expected points
    cmethod = {Clustering.single: clustering.single_linkage, Clustering.complete: clustering.complete_linkage, Clustering.average: clustering.average_linkage}
    clusters = cmethod[args.c](dataset, args.t)
    new_dataset = rank_cluster_points(dataset, clusters)
    logger.info(f'Clustered dataset ({len(new_dataset)})')

    
    axs[1].plot(points[:,0], points[:,1])
    for x,y in new_dataset:
        
        axs[1].plot(x, y, marker='o', markersize=3, color='red')

    plt.show()

    # Store the dataset into a CSV
    #with open(expected_file, 'w') as f:
    #    writer = csv.writer(f)
    #    writer.writerows(dataset) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hand Pick dataset generator')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-c', type=Clustering, choices=list(Clustering), help='clustering metric', default='average')
    parser.add_argument('-t', type=float, help='clustering threshold', default=0.05)
    args = parser.parse_args()
    
    main(args)