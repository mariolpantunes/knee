# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import typing
import logging
import numpy as np
import knee.rdp as rdp
import knee.linear_fit as lf
import knee.convex_hull as ch
import knee.knee_ranking as ranking

logger = logging.getLogger(__name__)


def filter_corner_knees(points: np.ndarray, knees: np.ndarray, t:float = .33) -> np.ndarray:
    """
    Filter the left upper corner knees points.

    A left upper knee corner does not provide a significant improvement to be considered.
    The detection method relies on a three point rectangle fitting and overlap.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes
        t (float): overlap treshold (default 0.33)

    Returns:
        np.ndarray: the filtered knees
    """

    filtered_knees = []

    for i in range(len(knees)-1):
        try:
            idx = knees[i]
            p0, p1 ,p2 = points[idx-1:idx+2]

            corner0 = np.array([p0[0], p2[1]])
            amin, amax = ranking.rect(corner0, p1)
            bmin, bmax = ranking.rect(p0, p2)
            p = ranking.rect_overlap(amin, amax, bmin, bmax)
            
            if p < t:
                filtered_knees.append(idx)
        except Exception as e:
            logger.debug(f'Exception: {e}')
            logger.debug(f'Corner detection issue: {idx}')
            logger.debug(f'Points: {points}')
            logger.debug(f'Knees : {knees}')
            # in this case consider the candidate point valid
            filtered_knees.append(idx)
    
    # deal with the last knee
    if len(knees) > 0:
        idx = knees[-1]
        if len(points) - idx > 1:
            p0, p1 ,p2 = points[idx-1:idx+2]
            
            corner0 = np.array([p0[0], p2[1]])
            amin, amax = ranking.rect(corner0, p1)
            bmin, bmax = ranking.rect(p0, p2)
            p = ranking.rect_overlap(amin, amax, bmin, bmax)
           
            if p < t:
                filtered_knees.append(idx)
        else:
            filtered_knees.append(idx)

    return np.array(filtered_knees)


def filter_worst_knees(points: np.ndarray, knees: np.ndarray) -> np.ndarray:
    """
    Filter the worst knees points.

    A worst knee is a knee that is higher (at y axis) than a previous knee.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes

    Returns:
        np.ndarray: the filtered knees
    """
    
    if len(knees) <= 1:
        return knees
    else:
        filtered_knees = []

        filtered_knees.append(knees[0])
        h_min = points[knees[0]][1]

        for i in range(1, len(knees)):
            h = points[knees[i]][1]

            if h <= h_min:
                filtered_knees.append(knees[i])
                h_min = h

        return np.array(filtered_knees)


def filter_clustring(points: np.ndarray, knees: np.ndarray,
clustering: typing.Callable[[np.ndarray, float], np.ndarray], t: float = 0.01,
method: ranking.ClusterRanking = ranking.ClusterRanking.linear) -> np.ndarray:
    """
    Filter the knee points based on clustering.

    For each cluster a single point is selected based on the ranking.
    The ranking is computed based on the slope and the improvement (on the y axis).

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes
        clustering (typing.Callable[[np.ndarray, float]): the clustering function
        t (float): the threshold for merging (in percentage, default 0.01)
        method (ranking.ClusterRanking): represents the direction of the ranking within a cluster (default ranking.ClusterRanking.linear)

    Returns:
        np.ndarray: the filtered knees
    """
    if method is ranking.ClusterRanking.hull:
        hull = ch.graham_scan_lower(points)

    if len(knees) <= 1:
        return knees
    else:
        knee_points = points[knees]
        clusters = clustering(knee_points, t)
        max_cluster = clusters.max()
        filtered_knees = []
        for i in range(0, max_cluster+1):
            current_cluster = knees[clusters == i]

            logger.info(f'Cluster {i} with {len(current_cluster)} elements')

            if len(current_cluster) > 1:
                rankings = ranking.cluster_ranking(points, current_cluster, method)
                idx = np.argmax(rankings)
                best_knee = knees[clusters == i][idx]
            else:
                best_knee = knees[clusters == i][0]
            filtered_knees.append(best_knee)

        return np.array(filtered_knees)


def add_points_even(points: np.ndarray, points_reduced: np.ndarray, knees: np.ndarray, removed:np.ndarray, tx:float=0.05, ty:float=0.05, extremes:bool=False) -> np.ndarray:
    """
    Add evenly spaced points between knees points.

    Whenever a smooth segment between two knew points are
    further away than tx (on the X-axis) and ty (on the Y axis),
    even spaced points are added to the result.
    This function will map the knees (in RDP space) into 
    the space of the complete set of points.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        points_reduced (np.ndarray): numpy array with the points (x, y) (simplified by RDP)
        knees (np.ndarray): knees indexes
        removed (np.ndarray): the points that were removed
        tx (float): the threshold (X-axis) for adding points (in percentage, default 0.05)
        ty (float): the threshold (Y-axis) for adding points (in percentage, default 0.05)
        extremes (bool): if True adds the extreme points (firt and last) (default False)

    Returns:
        np.ndarray: the resulting knees (mapped into the complete set of points)
    """
    
    # compute the delta x and y for the complete trace
    max_x, max_y = points.max(axis=0)
    min_x, min_y = points.min(axis=0)
    dx = math.fabs(max_x - min_x)
    dy = math.fabs(max_y - min_y)
    
    # compute the candidates
    candidates = []

    # check between knees
    for i in range(1, len(points_reduced)):
        left = i-1
        right = i
        pdx = math.fabs(points_reduced[right][0] - points_reduced[left][0])/dx
        pdy = math.fabs(points_reduced[right][1] - points_reduced[left][1])/dy
        if pdx > (2.0*tx) and pdy > ty:
            candidates.append(left)
            candidates.append(right)
    #print(f'candidates: {candidates}')
    
    # Map candidates into the complete set of points
    candidates = np.array(candidates)
    candidates = rdp.mapping(candidates, points_reduced, removed)

    # new knees
    new_knees = []

    # Process candidates as pairs
    for i in range(0, len(candidates), 2):
        left = candidates[i]
        right = candidates[i+1]
        pdx = math.fabs(points[right][0] - points[left][0])/dx
        number_points = int(math.ceil(pdx/(2.0*tx)))
        inc = int((right-left)/number_points)
        idx = left
        for _ in range(number_points):
            idx = idx + inc
            new_knees.append(idx)
    
    # filter worst knees that may be added due in this function
    # but keep the detected knees
    #new_knees = filter_worst_knees(points, new_knees)

    # Map knees into the complete set of points
    knees = rdp.mapping(knees, points_reduced, removed)

    # Add extremes points to the output
    if extremes:
        extremes_idx = [0, len(points)-1]
        knees_idx = np.concatenate((knees, new_knees, extremes_idx))
    else:
        knees_idx = np.concatenate((knees, new_knees))
    
    # np.concatenate generates float array when one is empty (see https://github.com/numpy/numpy/issues/8878)
    knees_idx = knees_idx.astype(int)
    knees_idx = np.unique(knees_idx)
    knees_idx.sort()
    #return knees_idx
    return filter_worst_knees(points, knees_idx)

def add_points_even_knees(points: np.ndarray, knees: np.ndarray, tx:float=0.05, ty:float=0.05, extremes:bool=False) -> np.ndarray:
    """
    Add evenly spaced points between knees points (using knee as markers).

    Whenever the distance between two consequetive knees is greater than 
    tx (on the X-axis) and ty (on the Y axis), even spaced points are 
    added to the result.
    The knees have to be mapped in the complete set of points.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes
        tx (float): the threshold (X-axis) for adding points (in percentage, default 0.05)
        ty (float): the threshold (Y-axis) for adding points (in percentage, default 0.05)
        extremes (bool): if True adds the extreme points (firt and last) (default False)

    Returns:
        np.ndarray: the resulting knees
    """
    # new knees
    new_knees = []

    # compute the delta x and y for the complete trace
    max_x, max_y = points.max(axis=0)
    min_x, min_y = points.min(axis=0)
    dx = math.fabs(max_x - min_x)
    dy = math.fabs(max_y - min_y)
    
    # compute the candidates
    candidates = []

    # check top part
    left = 0
    right = knees[0]
    pdx = math.fabs(points[right][0] - points[left][0])/dx
    pdy = math.fabs(points[right][1] - points[left][1])/dy

    if pdx > (2.0*tx) and pdy > ty:
        candidates.append((left, right))
    
    # check between knees
    for i in range(1, len(knees)):
        left = knees[i-1]
        right = knees[i]
        pdx = math.fabs(points[right][0] - points[left][0])/dx
        pdy = math.fabs(points[right][1] - points[left][1])/dy
        if pdx > (2.0*tx) and pdy > ty:
            candidates.append((left, right))
    
    # check last part
    left = knees[-1]
    right = len(points)-1
    pdx = math.fabs(points[right][0] - points[left][0])/dx
    pdy = math.fabs(points[right][1] - points[left][1])/dy

    if pdx > (2.0*tx) and pdy > ty:
        candidates.append((left, right))

    # Process candidates as pairs
    for left, right in candidates:
        pdx = math.fabs(points[right][0] - points[left][0])/dx
        number_points = int(math.ceil(pdx/(2.0*tx)))
        inc = int((right-left)/number_points)
        idx = left
        for _ in range(number_points):
            idx = idx + inc
            new_knees.append(idx)

    # Add extremes points to the output
    if extremes:
        extremes_idx = [0, len(points)]
        knees_idx = np.concatenate((knees, new_knees, extremes_idx))
    else:
        knees_idx = np.concatenate((knees, new_knees))
    
    # np.concatenate generates float array when one is empty (see https://github.com/numpy/numpy/issues/8878)
    knees_idx = knees_idx.astype(int)
    knees_idx = np.unique(knees_idx)
    knees_idx.sort()
    # filter worst knees that may be added due in this function
    return filter_worst_knees(points, knees_idx)
