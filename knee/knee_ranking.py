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


import matplotlib.pyplot as plt


class ClusterRanking(enum.Enum):
    """
    Enum data type that represents the direction of the ranking within a cluster.
    """
    left = 'left'
    linear = 'linear'
    right = 'right'
    corner = 'corner'

    def __str__(self):
        return self.value


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


def smooth_ranking(x, y, knees, t):
    fit = [0]
    weights = [0]
    
    j = knees[0]
    peak = np.max(y[knees])

    for i in range(1, len(knees)-1):
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

    weights.append(0)
    weights = np.array(weights)
    fit.append(0)
    fit = np.array(fit)

    #max_weights = np.max(weights)
    # if max_weights != 0:
    #    weights = weights / max_weights

    sum_weights = np.sum(weights)
    if sum_weights != 0:
        weights = weights / sum_weights

    rankings = fit * weights

    return rankings


def corner_ranking(points: np.ndarray, knees: np.ndarray) -> np.ndarray:
    """
    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes

    Returns:
        np.ndarray: an array with the ranks of each value
    """
    rankings = []
    for i in range(len(knees)):
        idx = knees[i]
        p0, p1 ,p2 = points[idx-1:idx+2]
        logger.info(f'Knee {i}/{idx} with points [{p0}, {p1}, {p2}]')
        corner0 = np.array([p2[0], p0[1]])
        #corner1 = np.array([p1[0], p2[1]]) if p2[1] < p1[1] else np.array([p1[0], 2.0*p1[1]-p2[1]])
        amin, amax = rect(corner0, p1)
        bmin, bmax = rect(p0, p2)
        rankings.append(rect_overlap(amin, amax, bmin, bmax))
    logger.info(f'Overlap: {rankings}')
    rankings = np.array(rankings)
    logger.info(f'Ranking: {rankings}')

    # To delete
    #x = points[:, 0]
    #y = points[:, 1]
    #plt.plot(x, y)
    #plt.plot(x[knees], y[knees], 'r+')
    #plt.show()

    return rankings


def cluster_ranking(points: np.ndarray, knees: np.ndarray, t: ClusterRanking = ClusterRanking.linear) -> np.ndarray:
    """Computes the rank for a cluster of knees in a curve.

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
    # trivial cases
    if len(knees) == 0:
        return np.array([])
    elif len(knees) == 1:
        return np.array([1.0])
    else:
        if t is ClusterRanking.corner:
            rankings = corner_ranking(points, knees)
        else:
            x = points[:, 0]
            y = points[:, 1]
            rankings = smooth_ranking(x,y,knees,t)

        # Compute relative ranking
        rankings = rank(rankings)
        # Min Max normalization
        rankings = (rankings - np.min(rankings))/np.ptp(rankings)

        logger.info(f'Final ranking: {rankings}')

        return rankings
