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


#plt.style.use('seaborn-v0_8-paper')
plt.style.use('tableau-colorblind10')
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.figsize'] = (8, 4)
plt.rcParams['lines.linewidth'] = 2


def function(x):
    return 1/x


def make_curvature(f):
    fp = jax.vmap(jax.grad(f))
    fpp = jax.vmap(jax.grad(jax.grad(f)))
    return lambda x: fpp(x)/jnp.power((1.0+jnp.power(fp(x), 2.0)), 1.5)


def main():
    # Color Blind adjusted colors and markers
    colormap=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', 
    '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']
    markers=['o', '*', '.', 'x', '+', 's', 'd', 'h', 'v']
    lines=['-', ':', '--', '-.']

    x_axis_min = 0.0
    x_axis_max = 3.0
    x = jnp.arange(x_axis_min, x_axis_max, 0.1)
    y = function(x)
    
    # compute the curvature (using numerical methods and JAX)
    kf = make_curvature(function)
    k = kf(x)

    # find the max of kf
    bounds = np.asarray([(x_axis_min, x_axis_max)])
    max_kf, _ = pso.particle_swarm_optimization(lambda x: -kf(x), bounds, n_pop=15, n_iter=15, verbose=False)

    plt.plot(x,y, color = colormap[0], linestyle=lines[0])
    plt.plot(x,k, color = colormap[2], linestyle=lines[2])
    plt.axvline(x = max_kf[0], color = colormap[1], linestyle=lines[1], label = 'axvline - full height')
    plt.savefig('out/function_curvature.png', bbox_inches='tight', transparent=True)
    plt.savefig('out/function_curvature.pdf', bbox_inches='tight', transparent=True)
    plt.show()

if __name__ == '__main__':
    main()