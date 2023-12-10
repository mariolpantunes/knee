# coding: utf-8

'''
The following module provides a set of methods
used for post-processing knees. These filter were
designed to improve the quality of the knee candidates.
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

import math
import logging
import numpy as np
import knee.rdp as rdp
import knee.linear_fit as lf
import knee.convex_hull as ch
import knee.knee_ranking as kr


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

    for i in range(len(knees)):
        idx = knees[i]
        if idx-1 >=0 and idx+1 < len(points):
            p0, p1 ,p2 = points[idx-1:idx+2]
            corner0 = np.array([p0[0], p2[1]])
            amin, amax = kr.rect(corner0, p1)
            bmin, bmax = kr.rect(p0, p2)
            p = kr.rect_overlap(amin, amax, bmin, bmax)
            
            if p < t:
                filtered_knees.append(idx)
        else:
            filtered_knees.append(idx)

    return np.array(filtered_knees)


def select_corner_knees(points: np.ndarray, knees: np.ndarray, t:float = .33) -> np.ndarray:
    """
    Detect and keep the left upper corner knees points.

    The detection method relies on a three point rectangle fitting and overlap.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes
        t (float): overlap treshold (default 0.33)

    Returns:
        np.ndarray: the filtered knees
    """

    filtered_knees = []

    for i in range(len(knees)):
        idx = knees[i]
        if idx-1 >=0 and idx+1 < len(points):
            p0, p1 ,p2 = points[idx-1:idx+2]
            corner0 = np.array([p0[0], p2[1]])
            amin, amax = kr.rect(corner0, p1)
            bmin, bmax = kr.rect(p0, p2)
            p = kr.rect_overlap(amin, amax, bmin, bmax)
            
            if p >= t:
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


def filter_clusters(points: np.ndarray, knees: np.ndarray,
clustering: callable, t: float = 0.01,
method: kr.ClusterRanking = kr.ClusterRanking.linear) -> np.ndarray:
    """
    Filter the knee points based on clustering.

    For each cluster a single point is selected based on the ranking.
    The ranking is computed based on the slope and the improvement (on the y axis).

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes
        clustering (callable): the clustering function
        t (float): the threshold for merging (in percentage, default 0.01)
        method (ranking.ClusterRanking): represents the direction of the ranking within a cluster (default ranking.ClusterRanking.linear)

    Returns:
        np.ndarray: the filtered knees
    """
    if method is kr.ClusterRanking.hull:
        hull = ch.graham_scan_lower(points)
        logger.info(f'hull {len(hull)}')

    x = points[:, 0]
    y = points[:, 1]

    if len(knees) <= 1:
        return knees
    else:
        knee_points = points[knees]
        clusters = clustering(knee_points, t)

        max_cluster = clusters.max()
        filtered_knees = []
        for i in range(0, max_cluster+1):
            current_cluster = knees[clusters == i]
            #logger.info(f'Cluster {i} with {len(current_cluster)} elements')

            if len(current_cluster) > 1:
                if method is kr.ClusterRanking.hull:
                    # select the hull points that exist within the cluster
                    a, b = current_cluster[[0, -1]]
                    #logger.info(f'Bounds [{a}, {b}]')
                    idx = (hull>=a)*(hull<=b)
                    hull_within_cluster = hull[idx]
                    #logger.info(f'Hull (W\\C) {hull_within_cluster} ({len(hull_within_cluster)})')
                    # only consider clusters with at least a single hull point
                    rankings = np.zeros(len(current_cluster))
                    
                    if len(hull_within_cluster) > 1:
                        length = x[b+1] - x[a-1]
                        for cluster_idx in range(len(current_cluster)):
                            j = current_cluster[cluster_idx]
                            if j in hull_within_cluster:
                                length_l = (x[j] - x[a-1])/length
                                length_r = (x[b+1] - x[j])/length
                                left = points[a-1:j+1]
                                right = points[j:b+2]
                                coef_l = lf.linear_fit_points(left)
                                coef_r = lf.linear_fit_points(right)
                                #r_l = lf.linear_residuals(x[a-1:j+1], y[a-1:j+1], coef_l)
                                #r_r = lf.linear_residuals(x[j:b+2], y[j:b+2], coef_r)
                                #r_l = lf.rmse_points(left, coef_l)
                                #r_r = lf.rmse_points(right, coef_r)
                                
                                r_l = np.sum(lf.shortest_distance_points(left, left[0], left[-1]))
                                r_r = np.sum(lf.shortest_distance_points(right, right[0], right[-1]))
                                
                                current_error = r_l * length_l  + r_r * length_r
                                rankings[cluster_idx] = current_error
                            else:
                                rankings[cluster_idx] = -1.0
                        # replace all -1 with maximum distance
                        #logger.info(f'CHR {rankings}')
                        rankings[rankings<0] = np.amax(rankings)
                        rankings = kr.distance_to_similarity(rankings)
                        #logger.info(f'CHRF {rankings}')
                    elif len(hull_within_cluster) == 1:
                        for cluster_idx in range(len(current_cluster)):
                            j = current_cluster[cluster_idx]
                            if j in hull_within_cluster:
                                rankings[cluster_idx] = 1.0
                    else:
                        rankings = None
                else:
                    rankings = kr.smooth_ranking(points, current_cluster, method)

                # Compute relative ranking
                if rankings is None:
                    best_knee = None
                else:
                    rankings = kr.rank(rankings)
                    #logger.info(f'Rankings {rankings}')
                    # Min Max normalization
                    #rankings = (rankings - np.min(rankings))/np.ptp(rankings)
                    idx = np.argmax(rankings)
                    best_knee = knees[clusters == i][idx]
            else:
                if method is kr.ClusterRanking.hull:
                    knee = knees[clusters == i][0]
                    if knee in hull:
                        best_knee = knee
                    else:
                        best_knee = None
                else:
                    best_knee = knees[clusters == i][0]
            
            if best_knee is not None:
                filtered_knees.append(best_knee)
                """# plot clusters within the points
                plt.plot(x, y)
                plt.plot(x[current_cluster], y[current_cluster], 'ro')
                if method is kr.ClusterRanking.hull:
                    plt.plot(x[hull], y[hull], 'g+')
                plt.plot(x[best_knee], y[best_knee], 'yx')
                plt.show()"""

        return np.array(filtered_knees)


def filter_clusters_corners(points: np.ndarray, knees: np.ndarray, clustering: callable, t: float = 0.01) -> np.ndarray:
    """
    This methods finds and removes corner points.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes
        clustering (callable): the clustering function
        t (float): the threshold for merging (in percentage, default 0.01)
    
    Returns:
        np.ndarray: the filtered knees

    """
    knee_points = points[knees]
    clusters = clustering(knee_points, t)

    max_cluster = clusters.max()
    filtered_knees = []
    for i in range(0, max_cluster+1):
        current_cluster = knees[clusters == i]
        # Compute the rank for each corner point
        ranks = rank_corners_triangle(points, current_cluster)
        idx = np.argmax(ranks)
        best_knee = knees[clusters == i][idx]
        filtered_knees.append(best_knee)
    return np.array(filtered_knees)



def add_points_even(points: np.ndarray, reduced: np.ndarray, knees: np.ndarray, removed:np.ndarray, tx:float=0.05, ty:float=0.05, extremes:bool=False) -> np.ndarray:
    """
    Add evenly spaced points between knees points.

    Whenever a smooth segment between two knew points are
    further away than tx (on the X-axis) and ty (on the Y axis),
    even spaced points are added to the result.
    This function will map the knees (in RDP space) into 
    the space of the complete set of points.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        reduced (np.ndarray): numpy array with the index of the reduced points (x, y) (simplified by RDP)
        knees (np.ndarray): knees indexes
        removed (np.ndarray): the points that were removed
        tx (float): the threshold (X-axis) for adding points (in percentage, default 0.05)
        ty (float): the threshold (Y-axis) for adding points (in percentage, default 0.05)
        extremes (bool): if True adds the extreme points (firt and last) (default False)

    Returns:
        np.ndarray: the resulting knees (mapped into the complete set of points)
    """
    
    points_reduced = points[reduced]

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
    candidates = rdp.mapping(candidates, reduced, removed)

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
    knees = rdp.mapping(knees, reduced, removed)

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


def triangle_area(p: np.ndarray) -> float:
    """
    Given 3 points computes the triangle area.

    Args:
        p (np.ndarray): numpy array with 3 2D points (x, y)
    
    Returns:
        float: triange area
    """
    area = 0.5 * (p[0][0]*(p[1][1]-p[2][1]) + p[1][0]*(p[2][1]-p[0][1]) + p[2][0]*(p[0][1]-p[1][1]))
    return area


def rank_corners_triangle(points: np.ndarray, knees: np.ndarray) -> np.ndarray:
    """
    Ranks knees based on the triangle fast heuristic.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes
    
    Returns:
        np.ndarray: the ranks
    """
    ranks = []

    for i in range(0, len(knees)):
        idx = [knees[i]-1, knees[i], knees[i]+1]
        pt = points[idx]
        #TODO: use the above function
        area = 0.5*((pt[1][0]-pt[0][0])*(pt[1][1]-pt[2][1]))
        ranks.append(area)

    return np.array(ranks)


def rank_corners(points: np.ndarray, knees: np.ndarray) -> np.ndarray:
    """
    Ranks knees based on their corners.
    
    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes
    
    Returns:
        np.ndarray: the ranks
    """
    ranks = []

    # compute the first rank
    d = points[knees[0]][0] - points[0][0]
    ranks.append(d)

    for i in range(1, len(knees)):
        d = points[knees[i]][0] - points[knees[i-1]][0]
        ranks.append(d)

    # return the ranks
    return np.array(ranks)