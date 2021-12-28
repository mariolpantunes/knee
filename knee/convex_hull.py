# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import numpy as np


def ccw(a:np.ndarray, b:np.ndarray, c:np.ndarray) -> float:
    """
    Check if three points make a counter-clockwise turn.

    Args:
        a (np.ndarray): numpy array with a single point (x, y)
        b (np.ndarray): numpy array with a single point (x, y)
        c (np.ndarray): numpy array with a single point (x, y)

    Returns:
        float: \\( \\gt 0\\) if counter-clockwise; \\( \\lt 0\\) if clockwise; \\( = 0\\) if collinear
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])


def sort_points(points:np.ndarray) -> np.ndarray:
    """
    Sort and array of points on their distance to the first point.

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        np.ndarray: the sorted points
    """
    p0 = points[0]
    return [p0] + sorted(points[1:], key=lambda p: (p0[1]-p[1])/(p0[0]-p[0]))


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
    sorted_points = sort_points(points)
    for p in sorted_points:
        # if we turn clockwise to reach this point, pop the last point from the stack, else, append this point to it.
        while len(stack) > 1 and ccw(stack[-2], stack[-1], p) <= 0:
            stack.pop()
        stack.append(p)
    # the stack is now a representation of the convex hull, return it.
    # convert the points into indexes
    x = points[:, 0]
    hull = [np.where(x == p[0])[0][0] for p in stack]
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