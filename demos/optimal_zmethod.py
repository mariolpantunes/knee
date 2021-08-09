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


import knee.rdp as rdp
import knee.zmethod as zmethod
import knee.evaluation as evaluation
import knee.optimization as opt
import knee.postprocessing as pp


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Global variable for optimization method
points = None
args = None
x_max = None
y_range = None
expected = None

cost_cache = {}
rdp_cache = {}


def compute_knee_points(r, dx, dy, dz, e):
    # RDP
    if r in rdp_cache:
        points_reduced, removed = rdp_cache[r]
    else:
        points_reduced, removed = rdp.rdp(points, r)
        rdp_cache[r] = (points_reduced, removed)

    ## Knee detection code ##
    knees = zmethod.knees(points_reduced, dx=dx, dy=dy, dz=dz, x_max=x_max, y_range=y_range)
    knees = knees[knees>0]
    knees = pp.add_points_even(points, points_reduced, knees, removed, tx=e, ty=e)

    return knees


def objective(p):
    # Round input parameters 
    r = round(p[0]*100.0)/100.0
    dx = round(p[1]*100.0)/100.0
    dy = round(p[2]*100.0)/100.0
    dz = round(p[3]*100.0)/100.0
    e = round(p[4]*100.0)/100.0

    # Check if cache already has these values
    cost = float('inf')
    if (r,dx, dy, dz, e) in cost_cache:
        cost = cost_cache[(r,dx, dy, dz, e)]
    else:
        knees = compute_knee_points(r, dx, dy, dz, e)
        
        # penalize solutions with a single knee
        if len(knees) == 1:
            cost = float('inf')
        else:
            cost = evaluation.rmspe(points, knees, expected, evaluation.Strategy.knees)
        cost_cache[(r,dx, dy, dz, e)] = cost

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
    c1 = np.array([p1[0], p2[1], p1[2], p2[3], p1[4]])
    c2 = np.array([p2[0], p1[1], p2[2], p1[3], p2[4]])
    return [c1, c2]


# mutation operator
def mutation(candidate, r_mut, bounds):
    if np.random.rand() < r_mut:
        solution = opt.get_random_solution(bounds)
        candidate[0] = solution[0]
        candidate[1] = solution[1]
        candidate[2] = solution[2]
        candidate[3] = solution[3]
        candidate[4] = solution[4]


def main(args):
    # Get the points
    global points 
    points = np.genfromtxt(args.i, delimiter=',')

    # Get original x_max and y_ranges
    global x_max
    x_max = [max(x) for x in zip(*points)][0]
    global y_range
    y_range = [[max(y),min(y)] for y in zip(*points)][1]

    # Get the expected values
    global expected
    dirname = os.path.dirname(args.i)
    filename = os.path.splitext(os.path.basename(args.i))[0]
    expected_file = os.path.join(os.path.normpath(dirname), f'{filename}_expected.csv')

    if os.path.exists(expected_file):
        with open(expected_file, 'r') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            expected = list(reader)
    else:
        expected = []
    expected = np.array(expected)

    # Run the Genetic Optimization
    bounds = np.asarray([[.85, .95], [.01, .1], [.01, .1], [.01, .1], [.01, .1]])
    best, score = opt.genetic_algorithm(objective, bounds, selection, crossover, mutation, n_iter=args.l, n_pop=args.p)

    # Round input parameters
    r = round(best[0]*100.0)/100.0
    dx = round(best[1]*100.0)/100.0
    dy = round(best[2]*100.0)/100.0
    dz = round(best[3]*100.0)/100.0
    e = round(best[4]*100.0)/100.0
    logger.info('%s (%s, %s, %s, %s, %s) = %s', args.i, r, dx, dy, dz, e, score)
    
    ### Run z-method ###
    logger.info(f'MSE(knees)   MSE(exp)   Cost(tr)   Cost(kn) RMSPE(knees) RMPSE(exp)')
    logger.info(f'-------------------------------------------------------------------')
    knees = compute_knee_points(r, dx, dy, dz, e)
    error_mse = evaluation.mse(points, knees, expected, evaluation.Strategy.knees)
    error_mse_exp = evaluation.mse(points, knees, expected, evaluation.Strategy.expected)
    error_rmspe = evaluation.rmspe(points, knees, expected, evaluation.Strategy.knees)
    error_rmspe_exp = evaluation.rmspe(points, knees, expected, evaluation.Strategy.expected)
    _,_,_,_,cost_trace = evaluation.accuracy_trace (points, knees)
    _,_,_,_,cost_knee = evaluation.accuracy_knee (points, knees)
    logger.info(f'{error_mse:10.2E} {error_mse_exp:10.2E} {cost_trace:10.2E} {cost_knee:10.2E} {error_rmspe:12.2E} {error_rmspe_exp:10.2E}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Z-Method Optimal Knee')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-p', type=int, help='population size', default=20)
    parser.add_argument('-l', type=int, help='number of loops (iterations)', default=200)
    args = parser.parse_args()
    
    main(args)