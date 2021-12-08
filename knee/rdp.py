# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
import math
import numpy as np
import knee.linear_fit as lf


import matplotlib.pyplot as plt


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


def point_distance(start: np.ndarray, end: np.ndarray):
    """
    """
    return math.sqrt(start[0]-end[0] + start[1] - end[1])

"""
def distance_point_line(pt: np.ndarray, start: np.ndarray, end: np.ndarray):
    #First, we need the length of the line segment.
    lineLength = point_distance(start, end)

    # if it's 0, the line is actually just a point.
    #if lineLength == 0:
    #    return point_distance(pt, start)
	
	t = ((p.x-i.x)*(j.x-i.x)+(p.y-i.y)*(j.y-i.y))/lineLength; 

	//t is very important. t is a number that essentially compares the individual coordinates
	//distances between the point and each point on the line.

	if(t<0){	//if t is less than 0, the point is behind i, and closest to i.
		return pointDistance(p,i);
	}	//if greater than 1, it's closest to j.
	if(t>1){
		return pointDistance(p,j);
	}
	return pointDistance(p, { x: i.x+t*(j.x-i.x),y: i.y+t*(j.y-i.y)});
	//this figure represents the point on the line that p is closest to.
"""

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
        #determination = lf.rmspe(x, y, coef)
        
        x = points[:, 0]
        y = points[:, 1]
        coef = np.polyfit(x,y,deg=1)
        f = np.poly1d(coef)
        y_hat = f(x)
        determination = np.sqrt(np.mean(np.square((y - y_hat) / y)))

        #determination = lf.linear_r2_points(points, coef)
    
       
        #plt.plot(x, y)
        #plt.plot(x, y_hat)
        logger.info(f'Determination = {determination}')
        #plt.show()

    #if determination < r:
    if determination >= r:
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
