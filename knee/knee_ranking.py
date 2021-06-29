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


class ClusterRanking(enum.Enum):
    """
    Enum data type that represents the direction of the ranking within a cluster.
    """
    left = 'left'
    linear = 'linear'
    right = 'right'

    def __str__(self):
        return self.value


logger = logging.getLogger(__name__)


def distance_to_similarity(array: np.ndarray) -> np.ndarray:
    """Converts an array of distances into an array of similarities.

    Args:
        array (np.ndarray): array with distances values

    Returns:
        np.ndarray: an array with similarity values
    """
    return max(array) - array


def rank(array: np.ndarray) -> np.ndarray:
    """Computes the rank of an array of values

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

        #print(f'Rankings = {rankings}')

        rankings = rank(rankings)
        # Min Max normalization
        if len(rankings) > 1:
            rankings = (rankings - np.min(rankings))/np.ptp(rankings)
        else:
            rankings = np.array([1.0])

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
        fit = [0]
        weights = [0]

        x = points[:, 0]
        y = points[:, 1]

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

        # Compute relative ranking
        rankings = rank(rankings)
        # Min Max normalization
        rankings = (rankings - np.min(rankings))/np.ptp(rankings)

        return rankings
