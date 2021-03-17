# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import os
import csv
import argparse
import numpy as np
import logging

import cProfile

import matplotlib.pyplot as plt
from knee.linear_fit import linear_fit, linear_residuals, linear_transform
from knee.rdp import perpendicular_distance

logger = logging.getLogger(__name__)


def convex(x, y):
    coef = linear_fit(x,y)
    y_hat = linear_transform(x, coef)
    return np.all(y-y_hat <= 0) or np.all(y-y_hat >= 0)

def monotonic(x):
    dx = np.diff(x)
    return np.all(dx <= 0) or np.all(dx >= 0)

#x = np.arange(0, 11)
#y = 1/(x+1)

x = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
y = np.array([1,1,0.7,0.538461538,0.4375,0.368421053,0.318181818,0.28,0.25,0.225806452,0.01,0.008264463,0.006944444,0.00591716,0.005102041,0.004444444,0.00390625,0.003460208,0.00308642,0.002770083])

#x = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
#y = np.array([1, 0.289589063,0.08385677,0.010416242,-0.010410937,-0.005965215,0.0119498,0.037481538,0.067447368,0.1,0.13401513,0.168784249,0.203849517,0.23891001,0.273766057,0.30828495,0.342379156,0.375992113,0.40908875,0.441649054])

#x = np.arange(0, 21)
#y = np.random.rand(21)


pr = cProfile.Profile()
pr.enable()

index = 1
length = x[-1] - x[0]
left_length = x[index] - x[0]
right_length = x[-1] - x[index]

errors = []
    
coef_left = linear_fit(x[0:index+1], y[0:index+1])
coef_right = linear_fit(x[index:], y[index:])
r_left = linear_residuals(x[0:index+1], y[0:index+1], coef_left)
r_rigth = linear_residuals(x[index:], y[index:], coef_right)
error = r_left*(left_length/length) + r_rigth*(right_length/length)

errors.append(error)

for i in range(index+1, len(x)-2):
    left_length = x[i] - x[0]
    right_length = x[-1] - x[i]
    
    i_coef_left = linear_fit(x[0:i+1], y[0:i+1])
    i_coef_right = linear_fit(x[i:], y[i:])
    r_left = linear_residuals(x[0:i+1], y[0:i+1], i_coef_left)
    r_rigth = linear_residuals(x[i:], y[i:], i_coef_right)
    current_error = r_left*(left_length/length) + r_rigth*(right_length/length)
    
    errors.append(current_error)

    if current_error < error:
        error = current_error
        index = i
        coef_left = i_coef_left
        coef_right = i_coef_right

pr.disable()
pr.print_stats()


x_error = x[1:-2]

points = np.transpose(np.stack((x,y)))
left = 0
right = len(x) - 1
d = perpendicular_distance(points, left, right)
middle = np.argmax(d)
middle_left = np.argmax(d[:middle])
middle_right = middle+1+np.argmax(d[middle+1:])



is_monotic = monotonic(y)
print(f'Is monotonic: {is_monotic}')

is_convex = convex(x,y)
print(f'Is convex: {is_convex}')

plt.plot(x,y, color='b')
plt.plot([x[left], x[index], x[right]], [y[left], y[index], y[right]], color='orange')
plt.plot(x_error, errors, color='r')
plt.axvline(x=middle_left, color='g')
plt.axvline(x=middle, color='black')
plt.axvline(x=middle_right, color='g')
plt.show()