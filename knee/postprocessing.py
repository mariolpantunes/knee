# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import typing
import logging
import numpy as np
import knee.linear_fit as lf
import knee.knee_ranking as ranking
import matplotlib.pyplot as plt



logger = logging.getLogger(__name__)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    """
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))


def rec_fit(p0, p1, p2) -> float:
    """
    """
    # normalize data
    #logger.info('%s %s %s', p0, p1, p2)
    x = np.array([p0, p1, p2])
    x_normed = (x - x.min(0)) / x.ptp(0)
    p0, p1, p2 = x_normed
    #logger.info('%s %s %s', p0, p1, p2)

    p1x, p1y = p1
    p0x, _ = p0
    _, p2y = p2
    a = p0 - p1
    b = np.array([p0x, p1y]) - p1
    v = cos_sim(a, b)

    b = (np.array([p1x, p2y]) - p1) if p2y < p1y else (np.array([p1x, p1y - p2y]) - p1)
    a = p2 - p1
    #logger.info('h %s %s', a, b)
    h = cos_sim(a, b)

    #logger.info('%s %s', v, h)

    return (v + h) / 2.0


def filter_corner_knees(points: np.ndarray, knees: np.ndarray, t:float = .99) -> np.ndarray:
    """
    """

    filtered_knees = []

    for i in range(0, len(knees)-1):
        idx = knees[i]
        p0, p1 ,p2 = points[idx-1:idx+2]
        r2 = rec_fit(p0, p1, p2)
        logger.info('knee %s, r2 %s (%s)', idx, r2, (r2 <= t))
        if r2 <= t:
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
