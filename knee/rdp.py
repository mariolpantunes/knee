# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
import numpy as np
import knee.linear_fit as lf


logger = logging.getLogger(__name__)


def perpendicular_distance(points: np.ndarray) -> np.ndarray:
    """
    Computes the perpendicular distance from the points to the 
    straight line defined by the first and last point.

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        np.ndarray: the perpendicular distances

    """
    return perpendicular_distance_index(points, 0, len(points) - 1)


def perpendicular_distance_index(points: np.ndarray, left: int, right: int) -> np.ndarray:
    """
    Computes the perpendicular distance from the points to the 
    straight line defined by the left and right point.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        left (int): the index of the left point
        right (int): the index of the right point

    Returns:
        np.ndarray: the perpendicular distances
    """
    return left + perpendicular_distance_points(points[left:right+1], points[left], points[right])


def perpendicular_distance_points(pt: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """
    Computes the perpendicular distance from the points to the 
    straight line defined by the left and right point.

    Args:
        pt (np.ndarray): numpy array with the points (x, y)
        start (np.ndarray): the left point
        end (np.ndarray): the right point

    Returns:
        np.ndarray: the perpendicular distances
    """
    return np.fabs(np.cross(end-start, pt-start)/np.linalg.norm(end-start))


def mapping(indexes: np.ndarray, points_reduced: np.ndarray, removed: np.ndarray) -> np.ndarray:
    """
    Computes the reverse of the RDP method.

    It maps the indexes on a simplified curve (using the rdp algorithm) into
    the indexes of the original points.
    This method assumes the indexes are sorted in ascending order.

    Args:
        indexes (np.ndarray): the indexes in the reduced space
        points_reduced (np.ndarray): the points that form the reduced space
        removed (np.ndarray): the points that were removed

    Returns:
        np.ndarray: the indexes in the original space
    """
    rv = []
    j = 0
    count = 0

    for i in indexes:
        value = points_reduced[i][0]
        #j = 0
        #idx = i
        while j < len(removed) and removed[j][0] < value:
            count += removed[j][1]
            j += 1
        idx = i + count
        rv.append(int(idx))

    return np.array(rv)


def rdp(points: np.ndarray, r: float = 0.9) -> tuple:
    """
    Ramer–Douglas–Peucker (RDP) algorithm.

    Is an algorithm that decimates a curve composed of line segments 
    to a similar curve with fewer points. This version uses the 
    coefficient of determination to decided whenever to keep or remove 
    a line segment.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        r(float): the coefficient of determination threshold (default 0.9)

    Returns:
        tuple: the reduced space, the points that were removed
    """

    if len(points) <= 2:
        determination = 1.0
    else:
        coef = lf.linear_fit_points(points)
        determination = lf.linear_r2_points(points, coef)

    if determination < r:
        d = perpendicular_distance_points(points, points[0], points[-1])
        index = np.argmax(d)

        left, left_points = rdp(points[0:index+1], r)
        right, right_points = rdp(points[index:len(points)], r)
        points_removed = np.concatenate((left_points, right_points), axis=0)
        return np.concatenate((left[0:len(left)-1], right)), points_removed
    else:
        rv = np.empty([2, 2])
        rv[0] = points[0]
        rv[1] = points[-1]
        points_removed = np.array([[points[0][0], len(points) - 2.0]])
        return rv, points_removed
