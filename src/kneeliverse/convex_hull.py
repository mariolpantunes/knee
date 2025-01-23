# coding: utf-8

'''
The following module provides optimized methods
for the computation of a convex hullfor 2D.
'''

__author__ = 'Mário Antunes'
__version__ = '1.0'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''


import functools
import numpy as np


def _ccw(a:np.ndarray, b:np.ndarray, c:np.ndarray) -> float:
    """
    Check if three points make a counter-clockwise turn.

    Args:
        a (np.ndarray): numpy array with a single point (x, y)
        b (np.ndarray): numpy array with a single point (x, y)
        c (np.ndarray): numpy array with a single point (x, y)

    Returns:
        float: $ \\gt 0$ if counter-clockwise; $ \\lt 0$ if clockwise; $ = 0$ if collinear
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])


def _dist_points(pi, pj) -> np.ndarray:
    """
    Compute the distance between two points ($p_i$, $p_j$)
    """
    return np.linalg.norm(pi - pj)


def _compare_points(p0, pi, pj) -> int:
    o = _ccw(p0, pi, pj)
    if o == 0:
        return -1 if _dist_points(p0, pj) >= _dist_points(p0, pi) else 1
    else:
        return 1 if o > 0 else -1


def _sort_points(points:np.ndarray) -> np.ndarray:
    """
    Sort the points based on the polar angle to the first point.
    
    The functions does not calculate the angle, instead, it calculate the relative orientation 
    of two points to find out which point makes the larger angle. 

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        np.ndarray: the sorted points
    """
    
    # find the smaller point and respective index
    p0 = min(points, key=lambda p: (p[0], p[1]))
    p0_idx = np.where(np.all(points==p0,axis=1))[0][0]
    
    # create a mask to select all points except the smaller point
    mask = np.full(len(points), True)
    mask[p0_idx] = False
    temp_list = points[mask]

    # return a copy of the points sorted
    return [p0] + sorted(temp_list, key=functools.cmp_to_key(lambda x, y: _compare_points(p0, x, y)))


def graham_scan(points:np.ndarray) -> np.ndarray:
    """
    Returns an array of indexes that make up the convex hull surrounding the points.
    
    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        np.ndarray: the indexes of the hull points
    """
    # convex_hull is a stack of points beginning with the leftmost point.
    stack = []
    sorted_points = _sort_points(points)

    if len(sorted_points) >= 3:
        stack.append(sorted_points[0])
        stack.append(sorted_points[1])
        stack.append(sorted_points[2])

    for i in range(3, len(sorted_points)):
        p = sorted_points[i]
        
        while _ccw(stack[-2], stack[-1], p) >= 0:
            stack.pop()
        stack.append(p)

    # the stack is now a representation of the convex hull, return it.
    # convert the points into indexes
    hull = [np.where(np.all(points==p,axis=1))[0][0] for p in stack]
    return np.array(hull)


def graham_scan_lower(points:np.ndarray) -> np.ndarray:
    """
    Returns an array of indexes that make up the lower convex hull.
    
    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        np.ndarray: the indexes of the hull points
    """
    #sorted_points = sort_points(points)
    # Add p0 and p1 to the stack
    stack = [0, 1]

    for i in range(2, len(points)):
        # if we turn clockwise to reach this point, pop the last point from the stack, else, append this point to it.
        while len(stack) > 1 and ccw(points[stack[-2]], points[stack[-1]], points[i]) <= 0:
            stack.pop()
        stack.append(i)
    # the stack is now a representation of the convex hull, return it.
    # convert the points into indexes
    #x = points[:, 0]
    #hull = [np.where(x == p[0])[0][0] for p in stack]
    return np.array(stack)


def graham_scan_upper(points:np.ndarray) -> np.ndarray:
    """
    Returns an array of indexes that make up the upper convex hull.
    
    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        np.ndarray: the indexes of the hull points
    """
    #sorted_points = sort_points(points)
    # Add p0 and p1 to the stack
    stack = [0, 1]

    for i in range(2, len(points)):
        # if we turn clockwise to reach this point, pop the last point from the stack, else, append this point to it.
        while len(stack) > 1 and ccw(points[i], points[stack[-1]], points[stack[-2]]) <= 0:
            stack.pop()
        stack.append(i)
    # the stack is now a representation of the convex hull, return it.
    # convert the points into indexes
    #x = points[:, 0]
    #hull = [np.where(x == p[0])[0][0] for p in stack]
    return np.array(stack)