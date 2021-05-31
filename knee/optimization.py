# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import logging
import numpy as np


logger = logging.getLogger(__name__)


# hill climbing local search algorithm
def hillclimbing(objective, bounds, n_iterations=200, step_size=.01):
    # min and max for each bound
    bounds_max = bounds.max(axis = 1)
    bounds_min = bounds.min(axis = 1)
    
    # generate an initial point
    solution = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    # evaluate the initial point
    solution_cost = objective(solution)
    # run the hill climb
    
    for i in range(n_iterations):
		# take a step
        candidate = solution + np.random.randn(len(bounds)) * step_size
        
        # Fix out of bounds value
        candidate = np.minimum(candidate, bounds_max)
        candidate = np.maximum(candidate, bounds_min)
        
        # evaluate candidate point
        candidte_cost = objective(candidate)
		# check if we should keep the new point
        
        if candidte_cost < solution_cost:
			# store the new point
            solution, solution_cost = candidate, candidte_cost
			# report progress
            #logger.info('>%d f(%s) = %.5f', i, solution, solution_cost)
    return [solution, solution_cost]

	
# simulated annealing algorithm
def simulated_annealing(objective, bounds, n_iterations=100, step_size=0.01, temp=10):
	# min and max for each bound
    bounds_max = bounds.max(axis = 1)
    bounds_min = bounds.min(axis = 1)
    # generate an initial point
    best = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# evaluate the initial point
    best_cost = objective(best)
	# current working solution
    curr, curr_cost = best, best_cost
	# run the algorithm
    for i in range(n_iterations):
		# take a step
        candidate = curr + np.random.randn(len(bounds)) * step_size
        # Fix out of bounds value
        candidate = np.minimum(candidate, bounds_max)
        candidate = np.maximum(candidate, bounds_min)
		# evaluate candidate point
        candidate_cost = objective(candidate)
		# check for new best solution
        #print(f'{i}/{n_iterations} -> {candidate} -> {candidate_cost}')
        if candidate_cost < best_cost:
			# store new best point
            best, best_eval = candidate, candidate_cost
			# report progress
            #logger.info('>%d f(%s) = %.5f' % (i, best, best_cost))
		# difference between candidate and current point evaluation
        diff = candidate_cost - curr_cost
        # calculate temperature for current epoch
        t = temp / float(i + 1)
        # calculate metropolis acceptance criterion
        metropolis = np.exp(-diff / t)
        # check if we should keep the new point
        if diff < 0 or np.random.rand() < metropolis:
            # store the new current point
            curr, curr_cost = candidate, candidate_cost
    return [best, best_eval]