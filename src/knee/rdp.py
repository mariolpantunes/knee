# coding: utf-8

'''
The following module provides a set of methods
used for curve simplification. It offers several
versions of the RDP algorhtm.
'''

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''

#import math
import enum
import logging
import numpy as np
import knee.linear_fit as lf
import knee.metrics as metrics
import knee.evaluation as evaluation


#import matplotlib.pyplot as plt

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


def compute_cost_coef(pt: np.ndarray, coef, cost: metrics.Metrics=metrics.Metrics.smape) -> float:
    """
    Computes the cost of fitting a linear function (with a given coefficient)
    in the points array.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        coef (tuple): the coefficients from the linear fit
        cost (lf.Linear_Metrics): the cost method used to evaluate a point set (default: metrics.Metrics.smape)

    Returns:
        float: the cost of fitting the linear function
    """
    methods = {metrics.Metrics.r2: lf.linear_r2_points,
    metrics.Metrics.rmspe: lf.rmspe_points,
    metrics.Metrics.rmsle: lf.rmsle_points,
    metrics.Metrics.smape: lf.smape_points,
    metrics.Metrics.rpd: lf.rpd_points}

    return methods[cost](pt, coef)


def rdp(points: np.ndarray, t: float = 0.01, distance: Distance = Distance.shortest, cost: metrics.Metrics = metrics.Metrics.smape) -> tuple:
    """
    Ramer–Douglas–Peucker (RDP) algorithm.

    Is an algorithm that decimates a curve composed of line segments to a similar curve with fewer points.
    This version uses different cost functions to decided whenever to keep or remove a line segment.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t (float): the coefficient of determination threshold (default 0.01)
        distance (Distance): the distance metric used to decide the split point (default: Distance.shortest)
        cost (metrics.Metrics): the cost method used to evaluate a point set (default: metrics.Metrics.smape)

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
    """
    Given an array of points and the reduced set it computes how many
    points were removed per segment.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        reduced (np.ndarray): numpy array with the reduced set of points
    
    Returns:
        np.ndarray: the points that were removed
    """
    removed = []

    left = reduced[0]
    for i in range(1, len(reduced)):
        right = reduced[i]
        pt = points[left:right+1]
        removed.append([left,len(pt)-2])
        left = right

    return np.array(removed)


def order_triangle(pt: np.ndarray, index:int, distance_points: callable) -> tuple:
    """
    Computes the triangle area of the left and right segments.
    The triangle area is a fast heuristic to estimate the how curve
    the left and right segment are.

    Args:
        pt (np.ndarray): numpy array with the points (x, y)
        index (int): the index that separates the left from the right segment
        distance_points (callable): methods that computes the distance between points

    Returns:
        tuple: the left and right triangle area
    """
    # compute the area of the triangles made from the farthest point
    base_left = np.linalg.norm(pt[0]-pt[index])
    pt_left = pt[0:index+1]
    height_left = distance_points(pt_left, pt_left[0], pt_left[-1]).max()
    left_tri_area = 0.5*base_left*height_left

    base_right = np.linalg.norm(pt[index]-pt[-1])
    pt_right = pt[index:]
    height_right = distance_points(pt_right, pt_right[0], pt_right[-1]).max()
    right_tri_area = 0.5*base_right*height_right

    return left_tri_area, right_tri_area


def order_area(pt: np.ndarray, index:int, distance_points:callable) -> tuple:
    """
    Computes the area of the left and right segments.
    The area is an heuristic to estimate the how curve
    the left and right segment are.

    Args:
        pt (np.ndarray): numpy array with the points (x, y)
        index (int): the index that separates the left from the right segment
        distance_points (callable): methods that computes the distance between points

    Returns:
        tuple: the left and right area
    """
    # compute the area using the distance function
    #pt_left = points[left:left+index+1]
    pt_left = pt[0:index+1]
    left_distance = distance_points(pt_left, pt_left[0], pt_left[-1])
    
    #pt_right = points[left+index:left+len(pt)]
    pt_right = pt[index:]
    right_distance = distance_points(pt_right, pt_right[0], pt_right[-1])

    left_area = np.sum(left_distance)
    right_area = np.sum(right_distance)
    
    return  left_area, right_area


def order_segment(pt: np.ndarray, index:int) -> tuple:
    """
    Computes the fitting error for the left and right segments.

    Args:
        pt (np.ndarray): numpy array with the points (x, y)
        index (int): the index that separates the left from the right segment
        
    Returns:
        tuple: the left and right fitting error
    """
    pt_left = pt[0:index+1]
    left_cost = lf.linear_fit_residuals_points(pt_left)
    
    pt_right = pt[index:]
    right_cost = lf.linear_fit_residuals_points(pt_right)
    
    return left_cost, right_cost


def _rdp_fixed(points: np.ndarray, length:int, distance_points:callable, 
order:Order, stack:list, reduced:list) -> list:
    """
    Main loop of the RDP fixed version.

    Not intended to be used as a single method.
    It is used internally on rdp_fixed and mp_grdp.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        length (int): the fixed length of reduced points 
        distance_points (callable): the distance function
        order (Order): the metric used to sort the segments
        stack (list): stack used to explore the points space
        reduced (list): set of reduced points

    Returns:
        list: the index of the reduced space
    """

    while length > 0 and stack:
        _, left, right = stack.pop()

        pt = points[left:right]
        d = distance_points(pt, pt[0], pt[-1])

        # fix issue when all distances are 0 (perfect fit)
        # changed to EPS for improve robustness
        #if d.sum() < np.finfo(float).eps:
        if np.all((d < np.finfo(float).eps)):
            index = int((len(d))/2)
        else:
            index = np.argmax(d)

        # add the relevant point to the reduced set
        reduced.append(left+index)

        if order is Order.triangle:
            left_cost, right_cost = order_triangle(pt, index, distance_points)
        elif order is Order.area:
            left_cost, right_cost = order_area(pt, index, distance_points)
        else:
            left_cost, right_cost = order_segment(pt, index)

        # Prevent the insertion of single point segments
        if (left+index+1) - (left) > 2:
            stack.append((left_cost, left, left+index+1))
        
        if (left+len(pt)) - (left+index) > 2:
            stack.append((right_cost, left+index, left+len(pt)))
        
        # Sort the stack based on the cost
        stack.sort(key=lambda t: t[0], reverse=False)
        length -= 1
    # sort the reduced set
    reduced.sort()
    return reduced


def rdp_fixed(points: np.ndarray, length:int=10, distance: Distance = Distance.shortest, order:Order=Order.segment) -> tuple:
    """
    Ramer–Douglas–Peucker (RDP) algorithm.

    Is an algorithm that decimates a curve composed of line segments to a similar curve with fewer points.
    This version reduces the number of points to a fixed length.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        length (int): the fixed length of reduced points (default: 10)
        distance (Distance): the distance metric used to decide the split point (default: Distance.shortest)
        order (Order): the metric used to sort the segments (default: Order.segment)

    Returns:
        tuple: the index of the reduced space, the points that were removed
    """
    stack = [(0, 0, len(points))]
    reduced = [0, len(points)-1]

    length -= 2

    # select the distance metric to be used
    distance_points = None
    if distance is Distance.shortest:
        distance_points = lf.shortest_distance_points
    elif distance is Distance.perpendicular:
        distance_points = lf.perpendicular_distance_points
    else:
        distance_points = lf.shortest_distance_points

    reduced = _rdp_fixed(points, length, distance_points, order, stack, reduced)
    reduced = np.array(reduced)
    return reduced, compute_removed_points(points, reduced)


def plot_frame(points, reduced, t):
    fig = plt.figure(figsize=(6, 6))
    x = points[:, 0]
    y = points[:, 1]
    plt.plot(x, y)
    points_reduced = points[reduced]
    x = points_reduced[:, 0]
    y = points_reduced[:, 1]
    plt.plot(x, y, marker='o', markersize=3)
    plt.savefig(f'./img/img_{t}.png', transparent = False,  facecolor = 'white')
    print(f'{t}')
    plt.close()


def _grdp(points:np.ndarray, t:float, cost:metrics.Metrics, order:Order, 
distance_points:callable, stack:list, reduced:list) -> tuple:
    """
    Main loop of the gRDP version.

    Not intended to be used as a single method.
    It is used internally on grdp and mp_grdp.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t (float): the coefficient of determination threshold
        cost (metrics.Metrics): the cost method used to evaluate a point set 
        order (Order): the metric used to sort the segments
        distance_points (callable): the distance function
        stack (list): stack used to explore the points space
        reduced (list): set of reduced points

    Returns:
        tuple: the index of the reduced space, stack
    """
    # Setup cache that is used to speedup the global cost computation
    cache = {}

    global_cost = evaluation.compute_global_cost(points, reduced, cost, cache)
    curved = global_cost < t if cost is metrics.Metrics.r2 else global_cost >= t

    ti = 0

    while curved and stack:
        _, left, right = stack.pop()
        pt = points[left:right]
        d = distance_points(pt, pt[0], pt[-1])

        # fix issue when all distances are 0 (perfect fit)
        # changed to EPS for improve robustness
        #if d.sum() < np.finfo(float).eps:
        if np.all((d < np.finfo(float).eps)):
            index = int((len(d))/2)
        else:
            index = np.argmax(d)
        
        # add the relevant point to the reduced set and sort
        reduced.append(left+index)
        reduced.sort()

        # compute the cost of the current solution
        global_cost = evaluation.compute_global_cost(points, reduced, cost, cache)
        curved = global_cost < t if cost is metrics.Metrics.r2 else global_cost >= t
        
        if order is Order.triangle:
            left_cost, right_cost = order_triangle(pt, index, distance_points)
        elif order is Order.area:
            left_cost, right_cost = order_area(pt, index, distance_points)
        else:
            left_cost, right_cost = order_segment(pt, index)
        
        # Prevent the insertion of single point segments
        if (left+index+1) - (left) > 2:
            stack.append((left_cost, left, left+index+1))
        
        if (left+len(pt)) - (left+index) > 2:
            stack.append((right_cost, left+index, left+len(pt)))
        
        # Sort the stack based on the cost
        stack.sort(key=lambda t: t[0], reverse=False)

        ## TO DELETE ##
        #plot_frame(points, reduced, ti)
        #ti = ti + 1 
        

    return reduced, stack


def grdp(points: np.ndarray, t: float = 0.01, distance: Distance = Distance.shortest, 
cost: metrics.Metrics = metrics.Metrics.smape, order:Order=Order.segment) -> tuple:
    """
    Global Ramer–Douglas–Peucker (RDP) algorithm.

    Is an algorithm that decimates a curve composed of line segments to a similar curve with fewer points.
    This version computes the global cost of reconstruction (instead of the cost of the current segment).
    It uses a cache to keep the cost of the segments that are not being explored right now.
    The exploration is based on the segment that has an overall higher reconstruction error.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t (float): the coefficient of determination threshold (default 0.01)
        distance (Distance): the distance metric used to decide the split point (default: Distance.shortest)
        cost (metrics.Metrics): the cost method used to evaluate a point set (default: metrics.Metrics.smape)
        order (Order): the metric used to sort the segments (default: Order.segment)
       
    Returns:
        tuple: the index of the reduced space, the points that were removed
    """

    stack = [(0, 0, len(points))]
    reduced = [0, len(points)-1]
    
    # select the distance metric to be used
    distance_points = None
    if distance is Distance.shortest:
        distance_points = lf.shortest_distance_points
    elif distance is Distance.perpendicular:
        distance_points = lf.perpendicular_distance_points
    else:
        distance_points = lf.shortest_distance_points

    reduced, _ = _grdp(points, t, cost, order, distance_points, stack, reduced)
    reduced = np.array(reduced)
    return reduced, compute_removed_points(points, reduced)


def mp_grdp(points: np.ndarray, t: float = 0.01, min_points:int = 10, distance: Distance = Distance.shortest,
cost: metrics.Metrics = metrics.Metrics.smape, order:Order=Order.segment) -> tuple:
    """
    MP gRDP (Min Points Global RDP)

    This version computes the gRDP with the given threshold.
    At the end, if the minimum number of points constraint is not satisfied
    it executes the rdp_fixed version.

    This method stores the reduced set and stack of the gRDP, meaning that
    the rdp_fixed method continues from the previous point onwards.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t (float): the coefficient of determination threshold (default 0.01)
        min_points (int): the minimal amount of points (default 10)
        distance (Distance): the distance metric used to decide the split point (default: Distance.shortest)
        cost (metrics.Metrics): the cost method used to evaluate a point set (default: metrics.Metrics.smape)
        order (Order): the metric used to sort the segments (default: Order.segment)
    
    Returns:
        tuple: the index of the reduced space, the points that were removed
    """

    stack = [(0, 0, len(points))]
    reduced = [0, len(points)-1]
    
    # select the distance metric to be used
    distance_points = None
    if distance is Distance.shortest:
        distance_points = lf.shortest_distance_points
    elif distance is Distance.perpendicular:
        distance_points = lf.perpendicular_distance_points
    else:
        distance_points = lf.shortest_distance_points

    reduced, stack = _grdp(points, t, cost, order, distance_points, stack, reduced)

    if len(reduced) >= min_points:
        reduced = np.array(reduced)
        return reduced, compute_removed_points(points, reduced)
    else:
        length = min_points - len(reduced)
        reduced = _rdp_fixed(points, length, distance_points, order, stack, reduced)
        reduced = np.array(reduced)
        return reduced, compute_removed_points(points, reduced)
    

def min_point_rdp(points: np.ndarray, t:list=[0.01, 0.001, 0.0001], min_points:int=10) -> tuple:
    """
    Minimal points RDP.

    Given a minimal amount of points, this version runs Global RDP with different threshold 
    values. The thresholds are sorted in decreasing order.
    The method returns as soon as a threshold value returns a minimal amount of points.
    If necessary the method will execute the fixed version of RDP to get the exact amount of points.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t (float): a list of coefficient of determination thresholds (default [0.01, 0.001, 0.0001])
        min_points (int): the minimal amount of points (default 10)

    Returns:
        tuple: the index of the reduced space, the points that were removed
    """    
    
    # sort the threshold in 
    t.sort(reverse=True)

    for current_t in t:
        reduced, removed = grdp(points, t=current_t)
        if len(reduced) >= min_points:
            return reduced, removed
    
    return rdp_fixed(points, min_points)
