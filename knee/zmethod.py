# coding: utf-8

__author__ = 'Tyler Estro'
__version__ = '0.1'
__email__ = 'testro@cs.stonybrook.edu'
__status__ = 'Development'


import numpy as np
import logging
import uts.gradient as grad
from uts.zscore import zscore_array


logger = logging.getLogger(__name__)


def map_index(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    """
    Maps the knee points into indexes.

    Args:
        a (np.ndarray): numpy array with the points (x)
        b (np.ndarray): numpy array with the knee points points (x)
    Returns:
        np.ndarray: The knee indexes
    """
    sort_idx = np.argsort(a)
    out = sort_idx[np.searchsorted(a, b, sorter=sort_idx)]
    return out


def knees(points:np.ndarray, dx:float=0.05, dy:float=0.05, dz:float=0.05, x_max:int=None, y_range:list=None) -> np.ndarray:
    """
    Given an array of points, it computes the knees.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        dx (float): % of max cache size between points (default 0.05)
        dy (float): % of max - min miss ratio between points (default 0.05)
        dz (float): amount we decrease outlier_z every iteration (default 0.05)
        x_max (int): max cache size of original (pre-RDP) MRC  (default None)
        y_max (list): [max, min] miss ratio of original (pre-RDP) MRC (default None)

    Returns:
        np.ndarray: The knee points on the curve
    """
    x = points[:, 0]
    rv = getPoints(points, dx, dy, dz, False, x_max, y_range)
    # convert x points into indexes:
    return map_index(x, np.array(rv))


def getPoints(points: np.ndarray, dx:float=0.05, dy:float=0.05, dz:float=0.05, plot:bool=False, x_max:int=None, y_range:list=None) -> np.ndarray:
    """
    Use our outlier method to find interesting points in an MRC.
    
    Args:
        points (np.ndarray): numpy array with the points (x, y)
        dx (float): % of max cache size between points (default 0.05)
        dy (float): % of max - min miss ratio between points (default 0.05)
        dz (float): amount we decrease outlier_z every iteration (default 0.05)
        plot (bool): set True if you want to return data useful for plotting
        x_max (int): max cache size of original (pre-RDP) MRC (default None)
        y_max (list): [max, min] miss ratio of original (pre-RDP) MRC (default None)

    Returns:
        list: list with the knees x coordinate
    """
    
    # in case we use RDP, we need the original MRC x/y ranges: x_max,y_range vars
    x_max = x_max if x_max else len(points)
    if y_range:
        y_max,y_min = y_range
    else:
        y_max,y_min = (points[:,1].max(),points[:,1].min())

    if len(points) < 4:
        logger.debug('pointSelector: < 4 unique requests in workload')
        return []

    if y_min == 1:
        logger.debug('pointSelector: workload completely random (dont bother caching)')
        return []

    # get absolute x and y distances
    x_width = max(1, int(x_max * dx))
    y_height = (y_max - y_min) * dy
    
    # get z-score
    x = points[:, 0]
    y = points[:, 1]
    yd2 = grad.csd(x, y)
    z_yd2 = zscore_array(x, yd2)
    min_zscore = min(z_yd2)
    # stack the 2nd derivative zscore with the points
    points = np.column_stack((points, z_yd2))

    # outlier_points holds our final selected points
    outlier_points = np.empty((0,2))
    
    # main loop. start with outliers >= 3 z-score
    outlier_z = 3
    while True:
    
        points_added = 0
        # candidate points have a zscore >= outlier_z
        candidates = points[points[:,2] >= outlier_z]
        #print('Candidates: ' + str(len(candidates)) + ' Points: ' + str(len(points)) + ' Outlier_Points: ' +
        #        str(len(outlier_points)) + ' Outlier_Z: '  + str(round(outlier_z,3)))
        
        if len(candidates) > 0:
            x_diff = np.argwhere(np.diff(candidates, axis=0)[:,0] >= x_width).flatten()
            if len(x_diff) == 0:
                outlier_best = candidates[np.argmin(candidates[:,1])] # best miss ratio in range
                if all(abs(outlier_best[1]-i) >= y_height for i in outlier_points[:,1]):
                    outlier_points = np.append(outlier_points, [[outlier_best[0], outlier_best[1]]], axis=0)
                    points = points[np.where(((points[:,0] <= (outlier_best[0] - x_width)) | (points[:,0] >= (outlier_best[0] + x_width))) & \
                                ((points[:,1] <= (outlier_best[1] - y_height)) | (points[:,1] >= (outlier_best[1] + y_height))))]
                    points_added += 1
            else:
                candidate_outliers = np.empty((0,3))
                x_diff = np.hstack(([0],x_diff,[len(candidates)-1]))
                
                # first create an array of candidate outliers
                for i in range(0, len(x_diff)-1):
                    # points in this form (0, 1) [1,2) ... [n,End)
                    if i == 0:
                        x_range = candidates[candidates[:,0] <= candidates[x_diff[i+1]][0]]
                    else:
                        x_range = candidates[(candidates[:,0] > candidates[x_diff[i]][0]) & (candidates[:,0] <= candidates[x_diff[i+1]][0])]
                    
                    outlier_best = x_range[np.argmin(x_range[:,1])] # point with best miss ratio in range
                    outlier_best_z = x_range[np.argmin(x_range[:,2])][2] # best z-score in range
                    outlier_best[2] = outlier_best_z
                    candidate_outliers = np.append(candidate_outliers, [outlier_best], axis=0)
                
                # sort all the candidate outliers by z-score in descending order
                candidate_outliers = candidate_outliers[np.argsort(candidate_outliers[:,2])][::-1]
                for outlier_best in candidate_outliers:
                    if all(abs(outlier_best[1]-i) >= y_height for i in outlier_points[:,1]):
                        outlier_points = np.append(outlier_points, [[outlier_best[0], outlier_best[1]]], axis=0)
                        points = points[np.where(((points[:,0] <= (outlier_best[0] - x_width)) | (points[:,0] >= (outlier_best[0] + x_width))) & \
                                    ((points[:,1] <= (outlier_best[1] - y_height)) | (points[:,1] >= (outlier_best[1] + y_height))))]
                        points_added += 1

        # terminating conditions (i think len(points) == 0 is all we need now)
        if len(points) == 0 or ((outlier_z <= min_zscore) and points_added == 0):
            break

        outlier_z -= dz

    # sweep through and points to avoid picking concavity issues
    outlier_min_mr = 1.0
    
    # convert to a dict so we can delete in-place
    outlier_points = {int(x[0]):x[1] for x in outlier_points}
    outlier_keys = list(sorted(outlier_points.keys()))
    for k in outlier_keys:
        if outlier_points[k] > outlier_min_mr:
            del outlier_points[k]
        else:
            outlier_min_mr = outlier_points[k]

    # returns sorted list of cache sizes
    if not plot:
        #return map_index(points, outlier_points)
        return np.array(list(sorted(outlier_points.keys())))
    else:
        return (outlier_points, z_yd2)
