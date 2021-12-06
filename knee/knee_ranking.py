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
import knee.evaluation as ev


logger = logging.getLogger(__name__)


class ClusterRanking(enum.Enum):
    """
    Enum data type that represents the direction of the ranking within a cluster.
    """
    left = 'left'
    linear = 'linear'
    right = 'right'
    corner = 'corner'
    hull = 'hull'

    def __str__(self):
        return self.value


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


def distance_to_similarity(array: np.ndarray) -> np.ndarray:
    """
    Converts an array of distances into an array of similarities.

    Args:
        array (np.ndarray): array with distances values

    Returns:
        np.ndarray: an array with similarity values
    """
    return max(array) - array


def rank(array: np.ndarray) -> np.ndarray:
    """
    Computes the rank of an array of values.

    Args:
        array (np.ndarray): array with values

    Returns:
        np.ndarray: an array with the ranks of each value
    """
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks


def slope_ranking(points: np.ndarray, knees: np.ndarray, t: float = 0.8) -> np.ndarray:
    """Computes the rank of a set of knees in a curve.

    The ranking is based on the slope of the left of the knee point.
    The left neighbourhood is computed based on the R2 metric.
    
    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes
        t (float): the R2 threshold for the neighbourhood (default 0.8)

    Returns:
        np.ndarray: an array with the ranks of each value
    """
    # corner case
    if len(knees) == 1.0:
        rankings = np.array([1.0])
    else:
        rankings = []

        x = points[:, 0]
        y = points[:, 1]

        _, _, slope = ev.get_neighbourhood(x, y, knees[0], 0, t)
        rankings.append(math.fabs(slope))

        for i in range(1, len(knees)):
            _, _, slope = ev.get_neighbourhood(x, y, knees[i], knees[i-1], t)
            rankings.append(math.fabs(slope))

        rankings = np.array(rankings)
        rankings = rank(rankings)
        # Min Max normalization
        if len(rankings) > 1:
            rankings = (rankings - np.min(rankings))/np.ptp(rankings)
        else:
            rankings = np.array([1.0])

    return rankings


def smooth_ranking(points: np.ndarray, knees: np.ndarray, t: ClusterRanking) -> np.ndarray:
    """
    Computes the rank for a cluster of knees in a curve.

    The ranking is a weighted raking based on the Y axis improvement and the
    slope/smoothed of the curve.
    This methods tries to find the best knee within a cluster of knees, this
    means that the boundaries for the computation are based on the cluster dimention.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes
        t (ClusterRanking): selects the direction where the curve must be smooth

    Returns:
        np.ndarray: an array with the ranks of each value
    """

    x = points[:, 0]
    y = points[:, 1]

    fit = []
    weights = []
    
    j = knees[0]
    peak = np.max(y[knees])

    for i in range(0, len(knees)):
        # R2 score
        r2 = 0
        if t is ClusterRanking.linear:
            r2_left = lf.r2(x[j:knees[i]+1], y[j:knees[i]+1])
            r2_right = lf.r2(
                x[knees[i]:knees[-1]], y[knees[i]:knees[-1]])
            r2 = (r2_left + r2_right) / 2.0
        elif t is ClusterRanking.left:
            r2 = lf.r2(x[j:knees[i]+1], y[j:knees[i]+1])
        else:
            r2 = lf.r2(x[knees[i]:knees[-1]], y[knees[i]:knees[-1]])
        fit.append(r2)

        # height of the segment
        d = math.fabs(peak - y[knees[i]])
        weights.append(d)

    #weights.append(0)
    weights = np.array(weights)
    #fit.append(0)
    fit = np.array(fit)

    #max_weights = np.max(weights)
    # if max_weights != 0:
    #    weights = weights / max_weights

    sum_weights = np.sum(weights)
    if sum_weights != 0:
        weights = weights / sum_weights

    #logger.info(f'Fit & Weights {fit} / {weights}')

    rankings = fit * weights

    #logger.info(f'Smooth Ranking {rankings}')

    return rankings


def corner_ranking(points: np.ndarray, knees: np.ndarray) -> np.ndarray:
    """
    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes

    Returns:
        np.ndarray: an array with the ranks of each value
    """
    corner_similarity = []
    heights_gain = []
    
    # compute a source point
    y = points[:, 1]
    peak = np.max(y[knees])


    logger.info(f'Peak {peak}')
    
    for i in range(len(knees)):
        idx = knees[i]
        if idx-1 >=0 and idx+1 < len(points):
            p0, p1 ,p2 = points[idx-1:idx+2]

            logger.info(f'Points [{p0}, {p1}, {p2}]')

            corner0 = np.array([p2[0], p0[1]])
            amin, amax = rect(corner0, p1)
            bmin, bmax = rect(p0, p2)

            overlap = rect_overlap(amin, amax, bmin, bmax)

            logger.info(f'Rect A {amin} {amax}')
            logger.info(f'Rect B {bmin} {bmax} = {overlap}')
        
            corner_similarity.append(overlap)
            heights_gain.append(peak-p1[1])
        else:
            corner_similarity.append(0)
            heights_gain.append(peak-points[idx][1])
    
    corner_similarity = np.array(corner_similarity)
    corner_similarity = corner_similarity/corner_similarity.sum()
    heights_gain = np.array(heights_gain)
    heights_gain = heights_gain/heights_gain.sum()

    rankings = corner_similarity*heights_gain

    logger.info(f'Corner {corner_similarity}')
    logger.info(f'Height {heights_gain}')
    logger.info(f'Ranking {rankings}')

    return rankings


def distances(point:np.ndarray, points:np.ndarray) -> np.ndarray:
    """
    """
    return np.sqrt(np.sum((points - point)**2, axis=1))


def convex_hull_ranking(points: np.ndarray, knees: np.ndarray, hull: np.ndarray) -> np.ndarray:
    slopes = []
    heights_gain = []
    
    # compute a source data point
    

    y = points[:, 1]
    peak = np.max(y[knees])

    #hull_points = points[hull]

    for i in range(len(hull)):
        idx = hull[i]
        # compute the slope of the this hull point
        p0, p1 = points[idx-1:idx+1]
        slope = math.fabs((p0[1] - p1[1])/(p0[0] - p1[0]))
        heights_gain.append(peak-p1[1])
        slopes.append(slope)
    
    slopes = np.array(slopes)
    slopes = slopes/slopes.sum()
    
    heights_gain = np.array(heights_gain)
    heights_gain = heights_gain/heights_gain.sum()

    rank = slopes * heights_gain

    logger.info(f'Slopes {slopes}')
    logger.info(f'Height {heights_gain}')
    logger.info(f'Rank {rank}')
    idx = np.argmax(rank)

    bp = points[hull[idx]]
    logger.info(f'BP {bp} ({idx})')
    dist = distances(bp, points[knees])
    return distance_to_similarity(dist)
