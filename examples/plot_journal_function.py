#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''


import jax
import logging
import numpy as np
import jax.numpy as jnp
import jax.scipy.optimize
import matplotlib.pyplot as plt

import pyBlindOpt.pso as pso

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['lines.linewidth'] = 2


def function(x):
    return 1/x


def make_curvature(f):
    fp = jax.vmap(jax.grad(f))
    fpp = jax.vmap(jax.grad(jax.grad(f)))
    return lambda x: fpp(x)/jnp.power((1.0+jnp.power(fp(x), 2.0)), 1.5)


def main():
    x_axis_min = 0.0
    x_axis_max = 3.0
    x = jnp.arange(x_axis_min, x_axis_max, 0.2)
    y = function(x)
    
    # compute the curvature (using numerical methods and JAX)
    kf = make_curvature(function)
    k = kf(x)

    # find the max of kf
    bounds = np.asarray([(x_axis_min, x_axis_max)])
    max_kf, _ = pso.particle_swarm_optimization(lambda x: -kf(x), bounds, n_pop=15, n_iter=15, verbose=False)

    plt.plot(x,y)
    plt.plot(x,k)
    plt.axvline(x = max_kf[0], color = 'b', label = 'axvline - full height')
    
    plt.show()

if __name__ == '__main__':
    main()