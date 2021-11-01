# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import numpy as np


def ccw(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])


def cross_product_orientation(a, b, c):
    """Returns the orientation of the set of points.
    >0 if x,y,z are clockwise, <0 if counterclockwise, 0 if co-linear.
    """

    return (b.get_y() - a.get_y()) * \
            (c.get_x() - a.get_x()) - \
            (b.get_x() - a.get_x()) * \
            (c.get_y() - a.get_y())


def graham_scan(points:np.ndarray):
    """Takes an array of points to be scanned.
    Returns an array of points that make up the convex hull surrounding the points passed in in point_array.
    """

    

    # convex_hull is a stack of points beginning with the leftmost point.
    convex_hull = []
    #sorted_points = sort_points(point_array)
    for p in points:
        # if we turn clockwise to reach this point, pop the last point from the stack, else, append this point to it.
        while len(convex_hull) > 1 and ccw(convex_hull[-2], convex_hull[-1], p) <= 0:
            convex_hull.pop()
        convex_hull.append(p)
    # the stack is now a representation of the convex hull, return it.
    return np.array(convex_hull)