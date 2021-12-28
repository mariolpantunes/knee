# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import enum
import math
import logging
import numpy as np
import knee.linear_fit as lf


logger = logging.getLogger(__name__)


class RDP_Cost(enum.Enum):
    """
    Enum that defines the cost for the RDP 
    """
    r2 = 'r2'
    rmspe = 'rmspe'

    def __str__(self):
        return self.value


class RDP_Distance(enum.Enum):
    """
    Enum that defines the distance method for the RDP 
    """
    perpendicular = 'perpendicular'
    shortest = 'shortest'

    def __str__(self):
        return self.value


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


def shortest_distance_points(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """
    Computes the shortest distance from the points to the 
    straight line defined by the left and right point.

    Args:
        pt (np.ndarray): numpy array with the points (x, y)
        start (np.ndarray): the left point
        end (np.ndarray): the right point

    Returns:
        np.ndarray: the perpendicular distances
    """

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


def rdp(points: np.ndarray, t: float = 0.01, cost: RDP_Cost = RDP_Cost.rmspe, distance: RDP_Distance = RDP_Distance.shortest) -> tuple:
    """
    Ramer–Douglas–Peucker (RDP) algorithm.

    Is an algorithm that decimates a curve composed of line segments to a similar curve with fewer points.
    This version uses different cost functions to decided whenever to keep or remove a line segment.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t (float): the coefficient of determination threshold (default 0.01)
        cost (RDP_Cost): the cost method used to evaluate a point set (default: RDP_Cost.rmspe)
        distance (RDP_Distance): the distance metric used to decide the split point (default: RDP_Distance.shortest)

    Returns:
        tuple: the index of the reduced space, the points that were removed
    """
    stack = [(0, len(points))]

    reduced = []
    removed = []

    # select the distance metric to be used
    distance_points = None
    if distance is RDP_Distance.shortest:
        distance_points = shortest_distance_points
    elif distance is RDP_Distance.perpendicular:
        distance_points = perpendicular_distance_points
    else:
        distance_points = shortest_distance_points

    while stack:
        left, right = stack.pop()
        pt = points[left:right]

        if len(pt) <= 2:
            if cost is RDP_Cost.rmspe:
                r = 0.0
            else:
                r = 1.0
        else:
            coef = lf.linear_fit_points(pt)
            if cost is RDP_Cost.rmspe:
                r = lf.rmspe_points(pt, coef)
            else:
                r = lf.linear_r2(pt, coef)

        curved = r >= t if cost is RDP_Cost.rmspe else r < t

        if curved:
            d = distance_points(pt, pt[0], pt[-1])
            index = np.argmax(d)
            stack.append((left+index, left+len(pt)))
            stack.append((left, left+index+1))
        else:
            reduced.append(left)
            removed.append([left, len(pt) - 2.0])

    reduced.append(len(points)-1)
    return np.array(reduced), np.array(removed)
