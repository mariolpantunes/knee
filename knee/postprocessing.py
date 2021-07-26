# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import typing
import logging
import numpy as np
import uts.gradient as grad
#import knee.linear_fit as lf
import knee.knee_ranking as ranking
import knee.linear_fit as lf
import knee.rdp as rdp
#import matplotlib.pyplot as plt
from plot import plot_knees_candidates

logger = logging.getLogger(__name__)


def rect_overlap(amin: np.ndarray, amax: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> float:
    """
    Computes the percentage of the overlap for two rectangles.

    Args:
        amin (np.ndarray): the low point in rectangle A
        amax (np.ndarray): the high point in rectangle A
        bmin (np.ndarray): the low point in rectangle B
        bmax (np.ndarray): the high point in rectangle B

    Returns:
        float: percentage of the overlap of two rectangles
    """
    #logger.info('%s %s %s %s', amin, amax, bmin, bmax)
    dx = max(0.0, min(amax[0], bmax[0]) - max(amin[0], bmin[0]))
    dy = max(0.0, min(amax[1], bmax[1]) - max(amin[1], bmin[1]))
    #logger.info('dx %s dy %s', dx, dy)
    overlap = dx * dy
    #logger.info('overlap = %s', overlap)
    if overlap > 0.0:
        a = np.abs(amax-amin)
        b = np.abs(bmax-bmin)
        total_area = a[0]*a[1] + b[0]*b[1] - overlap
        #print(f'overlap area = {overlap} total area =  {total_area}')
        return overlap / total_area
    else:
        return 0.0


def rect(p1: np.ndarray, p2: np.ndarray) -> tuple:
    """
    Creates the low and high rectangle coordinates from 2 points.

    Args:
        p1 (np.ndarray): one of the points in the rectangle
        p2 (np.ndarray): one of the points in the rectangle

    Returns:
        tuple: tuple with two points (low and high)
    """
    p1x, p1y = p1
    p2x, p2y = p2
    return np.array([min(p1x, p2x), min(p1y, p2y)]), np.array([max(p1x, p2x), max(p1y, p2y)])


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

    for i in range(0, len(knees)-1):
        idx = knees[i]
        p0, p1 ,p2 = points[idx-1:idx+2]
        #print(f'{p0}, {p1}, {p2}')
        
        corner0 = np.array([p0[0], p1[1]])
        corner1 = np.array([p1[0], p2[1]]) if p2[1] < p1[1] else np.array([p1[0], 2.0*p1[1]-p2[1]])
        amin, amax = rect(corner0, corner1)
        #print(f'{amin}, {amax}')

        bmin, bmax = rect(p0, p2)
        #print(f'{bmin}, {bmax}')
        
        p = rect_overlap(amin, amax, bmin, bmax)
        #print(f'knee {idx}, p {p} ({p < t})')
        if p < t:
            filtered_knees.append(idx)
    
    filtered_knees.append(knees[-1])

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

    if len(knees) <= 1:
        return knees
    else:
        knee_points = points[knees]
        clusters = clustering(knee_points, t)
        max_cluster = clusters.max()
        filtered_knees = []
        for i in range(0, max_cluster+1):
            current_cluster = knees[clusters == i]

            if len(current_cluster) > 1:
                rankings = ranking.cluster_ranking(points, current_cluster, method)
                #logger.info('Rankings: %s', rankings)
                idx = np.argmax(rankings)
                best_knee = knees[clusters == i][idx]
            else:
                #logger.info('Rankings: [1.0]')
                best_knee = knees[clusters == i][0]
            filtered_knees.append(best_knee)

            # plot cluster
            # if plot:
            #    xpoints = points[:,0]
            #    ypoints = points[:,1]
            #    plt.plot(xpoints, ypoints)
            #    plt.plot(xpoints[current_cluster], ypoints[current_cluster], marker='x', markersize=3, color='green')
            #    plt.plot(xpoints[best_knee], ypoints[best_knee], marker='o', markersize=5, color='red')
            #    plt.show()
        return np.array(filtered_knees)


def add_points_even(points: np.ndarray, points_reduced: np.ndarray, knees: np.ndarray, removed:np.ndarray, plt, tx:float=0.05, ty:float=0.05) -> np.ndarray:
    """
    Add evenly spaced points between knees points.

    Whenever a smooth segment between two knew points are further away than tx (on the X-axis) and ty (on the Y axis), even spaced points are added to the result.
    This function will map the knees (in RDP space) into the space of the complete set of points.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        points_reduced (np.ndarray): numpy array with the points (x, y) (simplified by RDP)
        knees (np.ndarray): knees indexes (from the complete set of points)
        removed (np.ndarray): the points that were removed
        tx (float): the threshold (X-axis) for adding points (in percentage, default 0.05)
        ty (float): the threshold (Y-axis) for adding points (in percentage, default 0.05)

    Returns:
        np.ndarray: the resulting knees (mapped into the complete set of points)
    """
    # new knees
    new_knees = []

    # compute the delta x and y for the complete trace
    dx = math.fabs(points[-1][0] - points[0][0])
    dy = math.fabs(points[-1][1] - points[0][1])
    
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
    
    # Map candidates into the complete set of points
    candidates = np.array(candidates)
    candidates = rdp.mapping(candidates, points_reduced, removed)

    # Map knees into the complete set of points
    knees = rdp.mapping(knees, points_reduced, removed)

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

    knees_idx = np.concatenate((knees, new_knees))
    # np.concatenate generates float array when one is empty (see https://github.com/numpy/numpy/issues/8878)
    knees_idx = knees_idx.astype(int)
    knees_idx = np.unique(knees_idx)
    knees_idx.sort()

    return knees_idx
