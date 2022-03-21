# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import enum
import logging
import numpy as np
import knee.linear_fit as lf


logger = logging.getLogger(__name__)


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


def rdp(points: np.ndarray, t: float = 0.01, cost: lf.Linear_Metrics = lf.Linear_Metrics.rpd, distance: RDP_Distance = RDP_Distance.shortest) -> tuple:
    """
    Ramer–Douglas–Peucker (RDP) algorithm.

    Is an algorithm that decimates a curve composed of line segments to a similar curve with fewer points.
    This version uses different cost functions to decided whenever to keep or remove a line segment.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t (float): the coefficient of determination threshold (default 0.01)
        cost (lf.Linear_Metrics): the cost method used to evaluate a point set (default: lf.Linear_Metrics.rmspe)
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
            if cost is lf.Linear_Metrics.r2:
                r = 1.0
            else:
                r = 0.0
        else:
            coef = lf.linear_fit_points(pt)
            if cost is lf.Linear_Metrics.r2:
                r = lf.linear_r2_points(pt, coef)
            elif cost is lf.Linear_Metrics.rmspe:
                r = lf.rmspe_points(pt, coef)
            elif cost is lf.Linear_Metrics.rmsle:
                r = lf.rmsle_points(pt, coef)
            else:
                r = lf.rpd_points(pt, coef)

        curved = r < t if cost is lf.Linear_Metrics.r2 else r >= t

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


def rdp_fixed(points: np.ndarray, length:int, distance: RDP_Distance = RDP_Distance.shortest):
    """
    Ramer–Douglas–Peucker (RDP) algorithm.

    Is an algorithm that decimates a curve composed of line segments to a similar curve with fewer points.
    This version reduces the number of points to a fixed length.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        lenght (int): the fixed length of reduced points
        distance (RDP_Distance): the distance metric used to decide the split point (default: RDP_Distance.shortest)

    Returns:
        tuple: the index of the reduced space, the points that were removed
    """
    stack = [(1, 0, len(points))]

    reduced = []

    #TODO: trivial cases
    length -= 2

    # select the distance metric to be used
    distance_points = None
    if distance is RDP_Distance.shortest:
        distance_points = lf.shortest_distance_points
    elif distance is RDP_Distance.perpendicular:
        distance_points = lf.perpendicular_distance_points
    else:
        distance_points = lf.shortest_distance_points

    while length > 0:
        _, left, right = stack.pop()
        pt = points[left:right]

        d = distance_points(pt, pt[0], pt[-1])
        index = np.argmax(d)
        # add the relevant point to the reduced set
        reduced.append(left+index)
        # compute the cost of the left and right parts
        left_cost = np.max(distance_points(pt[0:index+1], pt[0], pt[index]))
        right_cost = np.max(distance_points(pt[index:len(pt)], pt[0], pt[-1]))
        # Add the points to the stack
        stack.append((right_cost, left+index, left+len(pt)))
        stack.append((left_cost, left, left+index+1))
        # Sort the stack based on the cost
        stack.sort(key=lambda t: t[0])
        length -= 1

    # add first and last points
    reduced.append(0)
    reduced.append(len(points)-1)

    # sort indexes
    reduced.sort()

    return np.array(reduced)


def compute_cost_sequence(points: np.ndarray, reduced, cost: lf.Linear_Metrics = lf.Linear_Metrics.rpd, distance: RDP_Distance = RDP_Distance.shortest):
    # sort indexes
    reduced.sort()

    # select the distance metric to be used
    distance_points = None
    if distance is RDP_Distance.shortest:
        distance_points = lf.shortest_distance_points
    elif distance is RDP_Distance.perpendicular:
        distance_points = lf.perpendicular_distance_points
    else:
        distance_points = lf.shortest_distance_points

    left=0
    for right in reduced:
        pt=points[left:right]
        coef = lf.linear_fit_points(pt)



def grdp(points: np.ndarray, t: float = 0.01, cost: lf.Linear_Metrics = lf.Linear_Metrics.rpd, distance: RDP_Distance = RDP_Distance.shortest) -> tuple:
    stack = [(0, len(points))]

    reduced = []
    removed = []

    

    curved = True

    while curved:
        _, left, right = stack.pop()
        pt = points[left:right]

        d = distance_points(pt, pt[0], pt[-1])
        index = np.argmax(d)
        # add the relevant point to the reduced set
        reduced.append(left+index)
        # compute the cost of the left and right parts
        left_cost = np.max(distance_points(pt[0:index+1], pt[0], pt[index]))
        right_cost = np.max(distance_points(pt[index:len(pt)], pt[0], pt[-1]))
        # Add the points to the stack
        stack.append((right_cost, left+index, left+len(pt)))
        stack.append((left_cost, left, left+index+1))
        # Sort the stack based on the cost
        stack.sort(key=lambda t: t[0])
        length -= 1

        # compute the cost of the current solution
        
        curved = r < t if cost is lf.Linear_Metrics.r2 else r >= t

    # add first and last points
    reduced.append(0)
    reduced.append(len(points)-1)

    # sort indexes
    reduced.sort()

    return np.array(reduced)
    
    
    
    
    while stack:
        left, right = stack.pop()
        pt = points[left:right]

        if len(pt) <= 2:
            if cost is lf.Linear_Metrics.r2:
                r = 1.0
            else:
                r = 0.0
        else:
            coef = lf.linear_fit_points(pt)
            if cost is lf.Linear_Metrics.r2:
                r = lf.linear_r2_points(pt, coef)
            elif cost is lf.Linear_Metrics.rmspe:
                r = lf.rmspe_points(pt, coef)
            elif cost is lf.Linear_Metrics.rmsle:
                r = lf.rmsle_points(pt, coef)
            else:
                r = lf.rpd_points(pt, coef)

        curved = r < t if cost is lf.Linear_Metrics.r2 else r >= t

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
