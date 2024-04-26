# coding: utf-8

'''
The following module provides knee detection method
based on Z-method algorithm.
'''

__author__ = 'Tyler Estro'
__version__ = '1.0'
__email__ = 'testro@cs.stonybrook.edu'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''


import enum
import math
import logging
import numpy as np
import kneeliverse.postprocessing as pp


import uts.gradient as grad
import uts.zscore as uzscore #import zscore_array
#import uts.thresholding as uthres #import isodata


logger = logging.getLogger(__name__)


class Outlier(enum.Enum):
    """
    """
    zscore = 'zscore'
    iqr = 'iqr'
    hampel = 'hampel'

    def __str__(self):
        return self.value


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


def knees2(points:np.ndarray, dx:float=0.05, dy:float=0.05, out:Outlier=Outlier.iqr):
    x = points[:, 0]
    y = points[:, 1]

    logger.info(f'Number of points {len(points)}')
    
    # get the step for the space contrains (BB)
    x_step = (max(x) - min (x))*dx
    y_step = (max(y) - min (y))*dy
    
    # get the z-score from the second derivative
    yd2 = grad.csd(x, y)

    if out is Outlier.iqr:
        # interquartile range method
        q1, q3 = np.percentile(yd2, [25, 75])
        iqr = q3 - q1
        outlier_z = q3 + (1.5 * iqr)
        candidates = [i for i in range(len(yd2)) if yd2[i] >= outlier_z]
    elif out is Outlier.hampel:
        # Hampel method
        med = np.median(yd2)
        t = np.abs(yd2-med)
        outlier_z = np.median(t) * 4.5
        candidates = [i for i in range(len(yd2)) if yd2[i] >= outlier_z]
    else:
        # Z-score
        z_yd2 = uzscore.zscore_array(x, yd2) # <-- TODO T-score or other
        outlier_z = np.median(z_yd2) # <-- TODO outher metric
        candidates = [i for i in range(len(z_yd2)) if z_yd2[i] >= outlier_z]
    
    logger.info(f'Candidates: {candidates}({len(candidates)})')
    
    # filter worst and corner points
    candidates = pp.filter_worst_knees(points, candidates)
    candidates = pp.filter_corner_knees(points, candidates, t=0.3)

    counter = 0
    done = False
    while not done:
        counter += 1
        logger.info(f'Round {counter}')
        logger.info(f'Current candidates {candidates}')

        best_candidates = []
        # cluster points based on dx and dy
        for i in candidates:
            c = points[i]
            n = [candidates[j] for j in range(len(candidates)) if (math.fabs(points[candidates[j],0] - c[0])<=x_step) and (math.fabs(points[candidates[j],1] - c[1])<=y_step)]
            logger.info(f'Candidate {c}/{i} neighbourhood {n}')

            if len(n) == 1 and n[0] == i:
                best_candidates.append(i)
                logger.info(f'found best knee...')
            elif len(n) > 1:
                # find the best candidate from the remaining list
                r = pp.rank_corners(points, n)
                logger.info(f'{n} -> rank {r}')
                if n[np.argmax(r)] == i:
                    best_candidates.append(i)
                    logger.info(f'found best knee...')
            else:
                logger.info(f'Ups...')
        logger.info(f'Candidates {candidates}({len(candidates)}) best candidates {best_candidates}({len(best_candidates)})')
        if np.array_equal(best_candidates, candidates):
            done = True
        else:
            candidates = best_candidates

    logger.info(f'Final list {candidates}')

    return np.array(candidates)


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
    
    #logger.info(f'X width {x_width} y height {y_height}')

    # get z-score
    x = points[:, 0]
    y = points[:, 1]
    yd2 = grad.csd(x, y)
    z_yd2 = uzscore.zscore_array(x, yd2)
    min_zscore = min(z_yd2)

    #logger.info(f'zscore [{min_zscore} {max(z_yd2)}]')

    # stack the 2nd derivative zscore with the points
    points = np.column_stack((points, z_yd2))

    # outlier_points holds our final selected points
    outlier_points = np.empty((0,2))

    #logger.info(f'outlier points {outlier_points}')
    
    # main loop. start with outliers >= 3 z-score
    # TODO: Magic number
    outlier_z = 3
    #logger.info(f' Initial outlier Z value: {outlier_z}')
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
                x_diff = np.hstack(([0], x_diff,[len(candidates)-1]))
                
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
        
        #logger.info(f'Outlier Z value: {outlier_z}')


    # TODO: what is this step
    # sweep through and points to avoid picking concavity issues
    outlier_min_mr = 1.0
    
    # convert to a dict so we can delete in-place
    #logger.info(f'Outlier points {outlier_points}')
    outlier_points = {int(x[0]):x[1] for x in outlier_points}
    #logger.info(f'Outlier points {outlier_points}')
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
