#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import os
import csv
import math
import argparse
import numpy as np
import logging

from enum import Enum
import knee.rdp as rdp

import matplotlib.pyplot as plt

import knee.zmethod as zmethod

import knee.evaluation as evaluation
from knee.knee_ranking import ClusterRanking
from plot import get_dimention, plot_lines_knees_ranking, plot_lines_knees
from knee.evaluation import accuracy_trace, accuracy_knee
from knee.optimization import hillclimbing, simulated_annealing, genetic_algorithm, get_random_solution
from knee.knee_ranking import rank, slope_ranking
from plot import plot_knees

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Global variable for optimization method
points = None
args = None
cost_cache = {}


class Clustering(Enum):
    single = 'single'
    complete = 'complete'
    average = 'average'

    def __str__(self):
        return self.value


class Accuracy(Enum):
    knee = 'knee'
    trace = 'trace'

    def __str__(self):
        return self.value


def compute_knee_points(dx, dy, dz):
    knees = zmethod.knees(points, dx, dy, dz)
    return knees


def objective(p):
    # Round input parameters 
    dx = round(p[0]*100.0)/100.0
    dy = round(p[1]*100.0)/100.0
    dz = round(p[2]*100.0)/100.0

    # Check if cache already has these values
    cost = float('inf')
    if (dx, dy, dz) in cost_cache:
        cost = cost_cache[(dx, dy, dz)]
    else:
        knees = compute_knee_points(dx, dy, dz)
        
        # penalize solutions with a single knee
        if len(knees) == 1:
            cost = float('inf')
        else:
            if args.a is Accuracy.knee:
                _, _, _, _, cost = accuracy_knee(points, knees)
            else:
                _, _, _, _, cost = accuracy_trace(points, knees)
        cost_cache[(dx, dy, dz)] = cost

    return cost


# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = np.random.randint(len(pop))
	for ix in np.random.randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    c1 = np.array([p1[0], p2[1], p1[2]])
    c2 = np.array([p2[0], p1[1], p2[2]])
    return [c1, c2]


# mutation operator
def mutation(candidate, r_mut, bounds):
    if np.random.rand() < r_mut:
        solution = get_random_solution(bounds)
        candidate[0] = solution[0]
        candidate[1] = solution[1]
        candidate[2] = solution[2]


def main(args):
    global points 
    points = np.genfromtxt(args.i, delimiter=',')

    bounds = np.asarray([[.01, .1], [.01, .1], [.01, .1]])
    best, score = genetic_algorithm(objective, bounds, selection, crossover, mutation)

    # Round input parameters 
    dx = round(best[0]*100.0)/100.0
    dy = round(best[1]*100.0)/100.0
    dz = round(best[2]*100.0)/100.0
    logger.info('%s (%s, %s, %s, %s) = %s', args.i, dx, dy, dz, args.a, score)

    knees = zmethod.knees(points, dx, dy, dz)
    
    if args.o is None:
        plot_knees(plt, points, knees, '')
        plt.show()
    else:
        plot_knees(plt, points, knees, args.o)
        plt.savefig(args.o)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Z-Score Optimal Knee')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-a', type=Accuracy, choices=list(Accuracy), default='trace')
    parser.add_argument('-o', type=str, help='output file')
    args = parser.parse_args()
    
    main(args)
