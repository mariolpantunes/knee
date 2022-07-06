# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import enum
import logging
import numpy as np
import knee.linear_fit as lf
import knee.metrics as metrics
import knee.evaluation as evaluation


logger = logging.getLogger(__name__)


class Distance(enum.Enum):
    """
    Enum that defines the distance method for the RDP 
    """
    perpendicular = 'perpendicular'
    shortest = 'shortest'

    def __str__(self):
        return self.value


class Order(enum.Enum):
    """
    Enum that defines the distance method for the RDP 
    """
    triangle = 'triangle'
    area = 'area'
    segment = 'segment'

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


def compute_cost_coef(pt: np.ndarray, coef, cost: metrics.Metrics):
    methods = {metrics.Metrics.r2: lf.linear_r2_points,
    metrics.Metrics.rmspe: lf.rmspe_points,
    metrics.Metrics.rmsle: lf.rmsle_points,
    metrics.Metrics.smape: lf.smape_points,
    metrics.Metrics.rpd: lf.rpd_points}

    return methods[cost](pt, coef)


def rdp(points: np.ndarray, t: float = 0.01, cost: metrics.Metrics = metrics.Metrics.rpd, distance: Distance = Distance.shortest) -> tuple:
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
    if distance is Distance.shortest:
        distance_points = lf.shortest_distance_points
    elif distance is Distance.perpendicular:
        distance_points = lf.perpendicular_distance_points
    else:
        distance_points = lf.shortest_distance_points

    while stack:
        left, right = stack.pop()
        pt = points[left:right]

        if len(pt) <= 2:
            if cost is metrics.Metrics.r2:
                r = 1.0
            else:
                r = 0.0
        else:
            coef = lf.linear_fit_points(pt)
            r = compute_cost_coef(pt, coef, cost)

        curved = r < t if cost is metrics.Metrics.r2 else r >= t

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


def compute_removed_points(points: np.ndarray, reduced: np.ndarray) -> np.ndarray:
    removed = []

    left = reduced[0]
    for i in range(1, len(reduced)):
        right = reduced[i]
        pt = points[left:right+1]
        removed.append([left,len(pt)-2])
        left = right

    return np.array(removed)


def rdp_fixed(points: np.ndarray, length:int, distance: Distance = Distance.shortest, order:Order=Order.triangle) -> tuple:
    """
    Ramer–Douglas–Peucker (RDP) algorithm.

    Is an algorithm that decimates a curve composed of line segments to a similar curve with fewer points.
    This version reduces the number of points to a fixed length.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        length (int): the fixed length of reduced points
        distance (RDP_Distance): the distance metric used to decide the split point (default: RDP_Distance.shortest)
        order (Order): the metric used to sort the segments (default: Order.triangle)

    Returns:
        tuple: the index of the reduced space, the points that were removed
    """
    stack = [(0, 0, len(points))]
    reduced = [0, len(points)-1]

    #TODO: trivial cases
    length -= 2

    # select the distance metric to be used
    distance_points = None
    if distance is Distance.shortest:
        distance_points = lf.shortest_distance_points
    elif distance is Distance.perpendicular:
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
        reduced.sort()

        if order is Order.triangle:
            # compute the area of the triangles made from the farthest point
            base_left = np.linalg.norm(pt[0]-pt[index])
            pt_left = points[left:left+index+1]
            height_left = distance_points(pt_left, pt_left[0], pt_left[-1]).max()
            left_tri_area = 0.5*base_left*height_left

            base_right = np.linalg.norm(pt[index]-pt[-1])
            pt_right = points[left+index:left+len(pt)]
            height_right = distance_points(pt_right, pt_right[0], pt_right[-1]).max()
            right_tri_area = 0.5*base_right*height_right

            stack.append((left_tri_area, left, left+index+1))
            stack.append((right_tri_area, left+index, left+len(pt)))
        elif order is Order.area:
            # compute the area using the distance function
            pt_left = points[left:left+index+1]
            left_distance = distance_points(pt_left, pt_left[0], pt_left[-1])
            
            pt_right = points[left+index:left+len(pt)]
            right_distance = distance_points(pt_right, pt_right[0], pt_right[-1])

            left_area = np.sum(left_distance)
            right_area = np.sum(right_distance)

            stack.append((left_area, left, left+index+1))
            stack.append((right_area, left+index, left+len(pt)))
        else:
            # the cost is based on the segment error
            cost_index = reduced.index(left+index) - 1
            # compute the cost of the current solution
            left_cost, right_cost = evaluation.compute_segment_cost(points, reduced, cost_index)

            stack.append((left_cost, left, left+index+1))
            stack.append((right_cost, left+index, left+len(pt)))
        
        # Sort the stack based on the cost
        stack.sort(key=lambda t: t[0], reverse=False)
        length -= 1

    reduced = np.array(reduced)
    return reduced, compute_removed_points(points, reduced)


def grdp(points: np.ndarray, t: float = 0.01, cost: metrics.Metrics = metrics.Metrics.rpd, order:Order=Order.triangle, distance: Distance = Distance.shortest) -> tuple:
    stack = [(0, 0, len(points))]
    reduced = [0, len(points)-1]
    
    # Setup cache that is used to speedup the global cost computation
    cache = {}
    #cache = None

    # select the distance metric to be used
    distance_points = None
    if distance is Distance.shortest:
        distance_points = lf.shortest_distance_points
    elif distance is Distance.perpendicular:
        distance_points = lf.perpendicular_distance_points
    else:
        distance_points = lf.shortest_distance_points

    global_cost = evaluation.compute_global_cost(points, reduced, cost, cache)
    curved = global_cost < t if cost is metrics.Metrics.r2 else global_cost >= t

    while curved:
        _, left, right = stack.pop()
        pt = points[left:right]

        d = distance_points(pt, pt[0], pt[-1])
        index = np.argmax(d)
        
        # add the relevant point to the reduced set and sort
        reduced.append(left+index)
        reduced.sort()

        # compute the cost of the current solution
        global_cost = evaluation.compute_global_cost(points, reduced, cost, cache)
        curved = global_cost < t if cost is metrics.Metrics.r2 else global_cost >= t
        
        if order is Order.triangle:
            # compute the area of the triangles made from the farthest point
            base_left = np.linalg.norm(pt[0]-pt[index])
            pt_left = points[left:left+index+1]
            height_left = distance_points(pt_left, pt_left[0], pt_left[-1]).max()
            left_tri_area = 0.5*base_left*height_left

            base_right = np.linalg.norm(pt[index]-pt[-1])
            pt_right = points[left+index:left+len(pt)]
            height_right = distance_points(pt_right, pt_right[0], pt_right[-1]).max()
            right_tri_area = 0.5*base_right*height_right

            stack.append((left_tri_area, left, left+index+1))
            stack.append((right_tri_area, left+index, left+len(pt)))
        elif order is Order.area:
            # compute the area using the distance function
            pt_left = points[left:left+index+1]
            left_distance = distance_points(pt_left, pt_left[0], pt_left[-1])
            
            pt_right = points[left+index:left+len(pt)]
            right_distance = distance_points(pt_right, pt_right[0], pt_right[-1])

            left_area = np.sum(left_distance)
            right_area = np.sum(right_distance)

            stack.append((left_area, left, left+index+1))
            stack.append((right_area, left+index, left+len(pt)))
        else:
            # the cost is based on the segment error
            cost_index = reduced.index(left+index) - 1
            # compute the cost of the current solution
            left_cost, right_cost = evaluation.compute_segment_cost(points, reduced, cost_index)

            stack.append((left_cost, left, left+index+1))
            stack.append((right_cost, left+index, left+len(pt)))
        
        # Sort the stack based on the cost
        stack.sort(key=lambda t: t[0], reverse=False)

    reduced = np.array(reduced)
    return reduced, compute_removed_points(points, reduced)
