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
    
    for _ in range(n_iterations):
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
def simulated_annealing(objective, bounds, n_iterations=200, step_size=0.01, temp=20):
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


def get_random_solution(bounds):
    solution = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    solution = np.minimum(solution, bounds.max(axis = 1))
    solution = np.maximum(solution, bounds.min(axis = 1))
    return solution


# genetic algorithm
def genetic_algorithm(objective, bounds, selection, crossover, mutation, n_iter=100, n_pop=10, r_cross=0.9, r_mut=0.2):
    # initial population of random bitstring
	pop = [get_random_solution(bounds) for _ in range(n_pop)]
	# keep track of best solution
	best, best_eval = 0, objective(pop[0])
	# enumerate generations
	for _ in range(n_iter):
		# evaluate all candidates in the population
		scores = [objective(c) for c in pop]
		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				#logger.info('>%d, new best f(%s) = %.3f' % (gen,  pop[i], scores[i]))
		# select parents
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut, bounds)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
	return [best, best_eval]