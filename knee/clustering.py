# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import logging
import numpy as np


logger = logging.getLogger(__name__)


def single_linkage(points: np.ndarray, t: float = 0.01) -> np.ndarray:
    """Computes the 1D clustering of the input points.

    Efficient implementation that uses a single pass to compute
    the clusters.
    Computes the single linkage clustering based only on the x axis:
    $$
        D(C_1, C_2) = \\min_{c_1  \\in C_1, c_2 \\in C_2} d(c_1, c_2)
    $$

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t (float): the threshold for merging (in percentage, default 0.01)

    Returns:
        np.ndarray: the clusters ids
    """
    clusters = []
    cluster_index = 0
    duration = points[-1, 0] - points[0, 0]

    # First Point is a cluster
    clusters.append(cluster_index)

    for i in range(1, len(points)):
        distance = math.fabs(points[i][0]-points[i-1][0])/duration
        if distance < t:
            clusters.append(cluster_index)
        else:
            cluster_index += 1
            clusters.append(cluster_index)

    return np.array(clusters)


def complete_linkage(points: np.ndarray, t: float = 0.01) -> np.ndarray:
    """Computes the 1D clustering of the input points.

    Efficient implementation that uses a single pass to compute
    the clusters.
    Computes the complete linkage clustering based only on the x axis:
    $$
        D(C_1, C_2) = \\max_{c_1  \\in C_1, c_2 \\in C_2} d(c_1, c_2)
    $$

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t (float): the threshold for merging (in percentage, default 0.01)

    Returns:
        np.ndarray: the clusters ids
    """
    clusters = []
    cluster_index = 0
    duration = points[-1, 0] - points[0, 0]

    # First Point is a cluster
    clusters.append(cluster_index)
    cluster_point_idx = 0

    for i in range(1, len(points)):
        distance = math.fabs(
            points[i][0]-points[cluster_point_idx][0])/duration
        if distance < t:
            clusters.append(cluster_index)
        else:
            cluster_index += 1
            clusters.append(cluster_index)
            cluster_point_idx = i

    return np.array(clusters)


def average_linkage(points: np.ndarray, t: float = 0.01) -> np.ndarray:
    """Computes the 1D clustering of the input points.

    Efficient implementation that uses a single pass to compute
    the clusters.
    Computes the average linkage clustering based only on the x axis:
    $$
        D(C_1, C_2) = \\frac{1}{|C_1|}\\sum_{c_1 \\in C_1}d(c_1, C_2)
    $$

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t (float): the threshold for merging (in percentage, default 0.01)

    Returns:
        np.ndarray: the clusters ids
    """
    clusters = []
    cluster_index = 0
    duration = points[-1, 0] - points[0, 0]

    # First Point is a cluster
    clusters.append(cluster_index)
    cluster_center = points[0, 0]
    cluster_size = 1

    for i in range(1, len(points)):
        distance = math.fabs(points[i][0] - cluster_center)/duration
        if distance < t:
            clusters.append(cluster_index)
            # Update center
            cluster_center = (cluster_size/(cluster_size+1)) * \
                cluster_center + (1/(cluster_size+1)) * points[i][0]
            cluster_size += 1
            #logger.info('Cluster Center %s (%s)', cluster_center, cluster_size)
        else:
            cluster_index += 1
            clusters.append(cluster_index)
            cluster_center = points[i, 0]
            cluster_size = 1
            #logger.info('Cluster Center %s (%s)', cluster_center, cluster_size)

    return np.array(clusters)
