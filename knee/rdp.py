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
        distance_points = lf.shortest_distance_points
    elif distance is RDP_Distance.perpendicular:
        distance_points = lf.perpendicular_distance_points
    else:
        distance_points = lf.shortest_distance_points

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
                r = lf.linear_r2_points(pt, coef)

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
