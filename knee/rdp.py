# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
import numpy as np
import knee.linear_fit as lf


logger = logging.getLogger(__name__)


def get_r2_points(points):
    x = points[:,0]
    y = points[:,1]
    return get_r2(x, y)


def get_r2(x, y):
    #r2 = (np.corrcoef(x[i:b+1], y[i:b+1])[0,1])**2.0
    if len(x) <= 2:
        return 1.0
    else:
        r2 = (np.corrcoef(x, y)[0,1])**2.0
        return r2


def straight_line(points, a, b, t=0.8):
    #corner cases
    if abs(a - b) <= 1:
        return a
    
    # setup
    x = points[:,0]
    y = points[:,1]

    scores = []

    for i in range(a, b):
        coef = lf.linear_fit(x[i:b+1], y[i:b+1])
        r2 = lf.linear_r2(x[i:b+1], y[i:b+1], coef)
        #r2 = get_r2(x[i:b+1], y[i:b+1])
        scores.append(r2)
    scores = np.array(scores)
    #print(f'Naive scores: {scores}')
    #idx = (np.abs(scores - t)).argmin()
    idx = np.argmax(scores > t)

    return a+idx


def perpendicular_distance(points):
    left = 0
    right = len(points) - 1
    return perpendicular_distance_index(points, left, right)


def perpendicular_distance_index(pt, left, right):
    points = pt[left:right+1]
    start = pt[left]
    stop = pt[right]
    return left + perpendicular_distance_points(points, start, stop)


def perpendicular_distance_points(pt, start, end):
    return np.fabs(np.cross(end-start,pt-start)/np.linalg.norm(end-start))


def mapping(indexes, points_reduced, removed):

    rv = []
    j = 0
    count = 0

    for i in indexes:
        value = points_reduced[i][0]
        #j = 0
        #idx = i
        while j < len(removed) and removed[j][0] < value:
            count += removed[j][1]
            j += 1
        idx = i + count
        rv.append(int(idx))

    return np.array(rv)


def rdp(points, r=0.9):
    coef = lf.linear_fit_points(points)
    determination = lf.linear_r2_points(points, coef)

    if determination < r :
        d = perpendicular_distance_points(points, points[0], points[-1])
        index = np.argmax(d)

        left, left_points = rdp(points[0:index+1], r)
        right, right_points = rdp(points[index:len(points)], r)
        points_removed = np.concatenate((left_points, right_points), axis=0)
        return np.concatenate((left[0:len(left)-1], right)), points_removed
    else:
        rv = np.empty([2,2])
        rv[0] = points[0]
        rv[1] = points[-1]
        points_removed = np.array([[points[0][0], len(points) - 2.0]])
        return rv, points_removed
