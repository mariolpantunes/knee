# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import numpy as np
import math


def perpendicular_distance(pt, start, end):
    d = end - start

    mag = math.hypot(d[0], d[1])
    if mag > 0.0:
        d /= mag
    
    pv = pt - start
    pvdot = np.dot(d, pv)

    a = pv - pvdot * d

    return math.hypot(a[0], a[1])

def rdp(points, r=0.95):
    dmax = 0
    index = 0
    end = len(points) - 1

    for i in range(1, end):
        d = perpendicular_distance(points[i], points[0], points[end])
        if d > dmax :
            index = i
            dmax = d

    m = (points[end][1]-points[0][1])/(points[end][0]-points[0][0])
    b = points[0][1]- (m * points[0][0])
    y = np.transpose(points)[1]
    yhat = np.empty(len(y))
    for i in range(0, end+1):
        yhat[i] = points[i][0]*m+b
    
    ybar = np.sum(y)/len(y)          
    ssreg = np.sum((y-yhat)**2)
    sstot = np.sum((y - ybar)**2)

    if sstot > 0.0:
        determination = 1.0 - (ssreg / sstot)
    else:
        determination = 1.0 - ssreg

    if determination < r :
        left = rdp(points[0:index+1], r)
        right = rdp(points[index:end+1], r)
        return np.concatenate((left[0:len(left)-1], right))
    else:
        rv = np.empty([2,2])
        rv[0] = points[0]
        rv[1] = points[end]
        return rv


#points = np.array([[0.0, 0.0], [1.0, 2.0], [1.2, 4.0], [2.3, 6], [2.9, 8], [5, 10]])
#sp = rdp(points)
#print(sp)