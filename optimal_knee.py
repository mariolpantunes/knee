# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import argparse
import numpy as np
import logging

from enum import Enum
from knee.evaluation import performance, performance_individual
from knee.optimization import hillclimbing, simulated_annealing, genetic_algorithm, get_random_solution
from knee.knee_ranking import rank, slope_ranking
from knee.postprocessing import filter_clustring, filter_worst_knees
import matplotlib.pyplot as plt
from knee.rdp import rdp, mapping
from knee.knee_ranking import ClusterRanking
import knee.clustering as clustering
from plot import plot_ranking


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Global variable for optimization method
points = None
points_cache = {}
cost_cache = {}


class Clustering(Enum):
    single = 'single'
    complete = 'complete'
    average = 'average'

    def __str__(self):
        return self.value


def postprocessing(points, knees, c=Clustering.average, t=0.1, m=ClusterRanking.left):
    knees = filter_worst_knees(points, knees)
    cmethod = {Clustering.single: clustering.single_linkage, Clustering.complete: clustering.complete_linkage, Clustering.average: clustering.average_linkage}
    current_knees = filter_clustring(points, knees, cmethod[c], t, m)
    return current_knees


def compute_knee_points(r, t):
    # Check if cache already has these values
    if r in points_cache:
        points_reduced, removed = points_cache[r]
    else:
        points_reduced, removed = rdp(points, r)
        points_cache[r] = (points_reduced, removed)
    
    knees = np.arange(1, len(points_reduced))
    filtered_knees = postprocessing(points_reduced, knees, t = t)
    
    return points_reduced, removed, filtered_knees


def objective(p):
    # Round input parameters 
    r = round(p[0]*100.0)/100.0
    t = round(p[1]*100.0)/100.0

    # Check if cache already has these values
    cost = float('inf')
    if (r,t) in cost_cache:
        cost = cost_cache[(r,t)]
    else:
        _, _, knees = compute_knee_points(r, t)
        #knees = mapping(knees, points_reduced, removed)
        #avg_x, _, _, _, p = performance(points, knees)
        #cost = avg_x / p
        
        # Check the performance on the reduced space
        points_reduced, _ = points_cache[r]
        # penalize solutions with a single knee
        if len(knees) == 1:
            cost = float('inf')
        else:
            _, _, cost = performance_individual(points_reduced, knees)
        cost_cache[(r,t)] = cost

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
    c1 = np.array([p1[0], p2[1]])
    c2 = np.array([p2[0], p1[1]])
    return [c1, c2]


# mutation operator
def mutation(candidate, r_mut, bounds):
    if np.random.rand() < r_mut:
        solution = get_random_solution(bounds)
        candidate[0] = solution[0]
        candidate[1] = solution[1]


def main(args):
    global points 
    points = np.genfromtxt(args.i, delimiter=',')

    bounds = np.asarray([[.9, .99], [0.01, 0.1]])
    best, score = genetic_algorithm(objective, bounds, selection, crossover, mutation)

    print(best)

    # Round input parameters 
    r = round(best[0]*100.0)/100.0
    t = round(best[1]*100.0)/100.0
    logger.info('%s (%s, %s) = %s', args.i, r, t, score)

    points_reduced, removed, knees = compute_knee_points(r, t)
    rankings = slope_ranking(points_reduced, knees)
    filtered_knees = mapping(knees, points_reduced, removed)
    #avg_x, avg_y, avg_s = performance(points, filtered_knees)
    #logger.info('%s %s %s', avg_x, avg_y, p)
    
    if args.o is None:
        plot_ranking(plt, points, filtered_knees, rankings, '')
        plt.show()
    else:
        plot_ranking(plt, points, filtered_knees, rankings, args.o)
        plt.savefig(args.o)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RDP Optimal Knee')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-o', type=str, help='output file')
    args = parser.parse_args()
    
    main(args)