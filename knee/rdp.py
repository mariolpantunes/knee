# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
import math
import numpy as np
import knee.linear_fit as lf


import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def perpendicular_distance(points: np.ndarray) -> np.ndarray:
    """
    Computes the perpendicular distance from the points to the 
    straight line defined by the first and last point.

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        np.ndarray: the perpendicular distances

    """
    return perpendicular_distance_index(points, 0, len(points) - 1)


def perpendicular_distance_index(points: np.ndarray, left: int, right: int) -> np.ndarray:
    """
    Computes the perpendicular distance from the points to the 
    straight line defined by the left and right point.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        left (int): the index of the left point
        right (int): the index of the right point

    Returns:
        np.ndarray: the perpendicular distances
    """
    return left + perpendicular_distance_points(points[left:right+1], points[left], points[right])


def perpendicular_distance_points(pt: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """
    Computes the perpendicular distance from the points to the 
    straight line defined by the left and right point.

    Args:
        pt (np.ndarray): numpy array with the points (x, y)
        start (np.ndarray): the left point
        end (np.ndarray): the right point

    Returns:
        np.ndarray: the perpendicular distances
    """
    return np.fabs(np.cross(end-start, pt-start)/np.linalg.norm(end-start))


def mapping(indexes: np.ndarray, reduced: np.ndarray, removed: np.ndarray, sorted: bool = True) -> np.ndarray:
    """
    Computes the reverse of the RDP method.

    It maps the indexes on a simplified curve (using the rdp algorithm) into
    the indexes of the original points.
    This method assumes the indexes are sorted in ascending order.

    Args:
        indexes (np.ndarray): the indexes in the reduced space
        reduced (np.ndarray): the points that form the reduced space
        removed (np.ndarray): the points that were removed
        sorted (bool): True if the removed array is already sorted

    Returns:
        np.ndarray: the indexes in the original space
    """

    if not sorted:
        sorted_removed = removed[np.argsort(removed[:, 0])]
    else:
        sorted_removed = removed

    rv = []
    j = 0
    count = 0

    for i in indexes:
        value = reduced[i]
        while j < len(sorted_removed) and sorted_removed[j][0] < value:
            count += sorted_removed[j][1]
            j += 1
        idx = i + count
        rv.append(int(idx))

    return np.array(rv)


def point_distance(start: np.ndarray, end: np.ndarray):
    """
    """
    return math.sqrt(np.sum(np.square(start-end)))


"""def distance_point_line(points: np.ndarray, start: np.ndarray, end: np.ndarray):
    # First, we need the length of the line segment.
    lineLength = point_distance(start, end)

    distances = []

    for pt in points:
        # if it's 0, the line is actually just a point.
        # if lineLength == 0:
        #    return point_distance(pt, start)
        t = ((pt[0]-start[0]) * (end[0] - start[0]) + (pt[1] - start[1]) * (end[1] - start[1]))/lineLength

        # t is very important. t is a number that essentially compares the
        # individual coordinates distances between the point and each point on the line.
        
        if t < 0:  # if t is less than 0, the point is behind i, and closest to i.
            dist = point_distance(pt, start)
        elif t > 1:  # if greater than 1, it's closest to j.
            dist = point_distance(pt, end)
        else:
            dist = point_distance(pt, start+t*(end-start))
        distances.append(dist)

    logger.info(f'Distances ({t}) = {np.array(distances)}')

    return np.array(distances)"""

def distance_point_line(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    
    # TODO for you: consider implementing @Eskapp's suggestions
    if np.all(a == b):
        return np.linalg.norm(p - a, axis=1)

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(p))])

    # perpendicular distance component, as before
    # note that for the 3D case these will be vectors
    c = np.cross(p - a, d)

    # use hypot for Pythagoras to improve accuracy
    return np.hypot(h, c)


def rdp(points: np.ndarray, t: float = 0.01) -> tuple:
    # Stack strucuture for the recursion
    stack = [(0, len(points))]

    reduced = []
    removed = []

    while stack:
        left, right = stack.pop()
        pt = points[left:right]

        if len(pt) <= 2:
            r = 0.0
        else:
            coef = lf.linear_fit_points(pt)
            r = lf.rmspe_points(pt, coef)

        if r >= t:
            #d = perpendicular_distance_points(pt, pt[0], pt[-1])
            #logger.info(f'PDP = {np.argmax(d)}')
            d = distance_point_line(pt, pt[0], pt[-1])
            #logger.info(f'SDP = {np.argmax(d)}')
            index = np.argmax(d)
            stack.append((left+index, left+len(pt)))
            stack.append((left, left+index+1))
        else:
            reduced.append(left)
            removed.append([left, len(pt) - 2.0])

    reduced.append(len(points)-1)
    return np.array(reduced), np.array(removed)
