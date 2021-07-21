# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import enum
import math
import logging
import numpy as np
import knee.linear_fit as lf


logger = logging.getLogger(__name__)


class Strategy(enum.Enum):
    """
    Enum data type that represents the strategy of MAE, MSE and RMSE
    """
    knees = 'knees'
    expected = 'expected'
    best = 'best'
    worst = 'worst'

    def __str__(self):
        return self.value


def get_neighbourhood_points(points: np.ndarray, a: int, b: int, t: float) -> tuple:
    """Get the neighbourhood (closest points) from a to b.

    The neighbourhood is defined as the longest straitgh line (defined by R2).

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        a (int): the initial point of the search
        b (int): the left limit of the search
        t (float): R2 threshold

    Returns:
        tuple: (neighbourhood index, r2, slope)
    """
    x = points[:, 0]
    y = points[:, 1]
    return get_neighbourhood(x, y, a, b, t)


def get_neighbourhood(x: np.ndarray, y: np.ndarray, a: int, b: int, t: float = 0.7) -> tuple:
    """Get the neighbourhood (closest points) from a to b.

    The neighbourhood is defined as the longest straitgh line (defined by R2).

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates
        a (int): the initial point of the search
        b (int): the left limit of the search
        t (float): R2 threshold

    Returns:
        tuple: (neighbourhood index, r2, slope)
    """
    
    r2 = 1.0
    i = a - 1
    _, slope = lf.linear_fit(x[i:a+1], y[i:a+1])

    while r2 > t and i > b:
        previous_res = (i, r2, slope)
        i -= 1
        coef = lf.linear_fit(x[i:a+1], y[i:a+1])
        r2 = lf.linear_r2(x[i:a+1], y[i:a+1], coef)
        _, slope = coef

    if r2 > t:
        return i, r2, slope
    else:
        return previous_res


def accuracy_knee(points: np.ndarray, knees: np.ndarray) -> tuple:
    """Compute the accuracy heuristic for a set of knees.

    The heuristic is based on the average distance of X and Y axis, the slope and the R2.
    In this version it is used the left neighbourhood of the knee.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes

    Returns:
        tuple: (average_x, average_y, average_slope, average_coeffients, cost)
    """
    x = points[:, 0]
    y = points[:, 1]

    total_x = math.fabs(x[-1] - x[0])
    total_y = math.fabs(y[-1] - y[0])

    distances_x = []
    distances_y = []
    slopes = []
    coeffients = []

    previous_knee = 0
    for i in range(0, len(knees)):
        idx, r2, slope = get_neighbourhood(x, y, knees[i], previous_knee)

        delta_x = x[idx] - x[knees[i]]
        delta_y = y[idx] - y[knees[i]]

        distances_x.append(math.fabs(delta_x))
        distances_y.append(math.fabs(delta_y))
        slopes.append(math.fabs(slope))
        coeffients.append(r2)

        previous_knee = knees[i]

    slopes = np.array(slopes)
    slopes = slopes/slopes.max()

    coeffients = np.array(coeffients)
    coeffients = coeffients/coeffients.max()

    distances_x = np.array(distances_x)/total_x
    distances_y = np.array(distances_y)/total_y
    average_x = np.average(distances_x)
    average_y = np.average(distances_y)

    average_slope = np.average(slopes)
    average_coeffients = np.average(coeffients)

    #p = slopes * distances_y * coeffients
    p = slopes * distances_y
    #cost = (average_x * average_y) / (average_slope)
    cost = average_x / np.average(p)

    return average_x, average_y, average_slope, average_coeffients, cost


def accuracy_trace(points: np.ndarray, knees: np.ndarray) -> tuple:
    """Compute the accuracy heuristic for a set of knees.

    The heuristic is based on the average distance of X and Y axis, the slope and the R2.
    In this version it is used the points from the current knee to the previous.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes

    Returns:
        tuple: (average_x, average_y, average_slope, average_coeffients, cost)
    """
    x = points[:, 0]
    y = points[:, 1]

    distances_x = []
    distances_y = []
    slopes = []
    coeffients = []

    total_x = math.fabs(x[-1] - x[0])
    total_y = math.fabs(y[-1] - y[0])

    previous_knee_x = x[knees[0]]
    previous_knee_y = y[knees[0]]

    delta_x = x[0] - previous_knee_x
    delta_y = y[0] - previous_knee_y
    distances_x.append(math.fabs(delta_x))
    distances_y.append(math.fabs(delta_y))

    coef = lf.linear_fit(x[0:knees[0]+1], y[0:knees[0]+1])
    r2 = lf.linear_r2(x[0:knees[0]+1], y[0:knees[0]+1], coef)
    coeffients.append(r2)
    _, slope = coef
    slopes.append(math.fabs(slope))

    for i in range(1, len(knees)):
        knee_x = x[knees[i]]
        knee_y = y[knees[i]]

        delta_x = previous_knee_x - knee_x
        delta_y = previous_knee_y - knee_y

        coef = lf.linear_fit(x[knees[i-1]:knees[i]+1],
                             y[knees[i-1]:knees[i]+1])
        r2 = lf.linear_r2(x[knees[i-1]:knees[i]+1],
                          y[knees[i-1]:knees[i]+1], coef)

        distances_x.append(math.fabs(delta_x))
        distances_y.append(math.fabs(delta_y))
        _, slope = coef
        slopes.append(math.fabs(slope))
        coeffients.append(r2)

        previous_knee_x = knee_x
        previous_knee_y = knee_y

    distances_x = np.array(distances_x)/total_x
    distances_y = np.array(distances_y)/total_y
    slopes = np.array(slopes)
    slopes = slopes/slopes.max()

    coeffients = np.array(coeffients)
    coeffients = coeffients/coeffients.max()

    coeffients[coeffients < 0] = 0.0
    p = slopes * distances_y * coeffients
    #p = slopes * distances_y

    average_x = np.average(distances_x)
    average_y = np.average(distances_y)
    average_slope = np.average(slopes)
    average_coeffients = np.average(coeffients)

    cost = average_x / np.average(p)

    return average_x, average_y, average_slope, average_coeffients, cost


def mae(points: np.ndarray, knees: np.ndarray, expected: np.ndarray, s: Strategy = Strategy.expected) -> float:
    """
    Estimates the worst case Mean Absolute Error (MAE) for the given
    knee and expected points.

    Suppports different size arrays, and estimates the MAE based 
    on the worst case.
    It uses the euclidean distance to find the closer points,
    and computes the error based on the closest point.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes
        expected (np.ndarray): numpy array with the expected knee points (x, y)
        s (Strategy): enum that controls the point matching (default Strategy.expected)

    Returns:
        float: the worst case MAE
    """
    # get the knee points
    knee_points = points[knees]

    error = 0.0
    
    if s is Strategy.knees:
        a = knee_points
        b = expected
    elif s is Strategy.expected:
        a = expected
        b = knee_points
    elif s is Strategy.best:
        if len(expected) <= len(knee_points):
            a = expected
            b = knee_points
        else:
            a = knee_points
            b = expected
    else:
        if len(expected) >= len(knee_points):
            a = expected
            b = knee_points
        else:
            a = knee_points
            b = expected

    for p in a:
        distances = np.linalg.norm(b-p, axis=1)
        idx = np.argmin(distances)
        error += np.sum(np.abs(p-b[idx]))

    return error / (len(a)*2.0)


def mse(points: np.ndarray, knees: np.ndarray, expected: np.ndarray, s: Strategy = Strategy.expected) -> float:
    """
    Estimates the worst case Mean Squared Error (MSE) for the given
    knee and expected points.

    Suppports different size arrays, and estimates the MSE based 
    on the worst case.
    It uses the euclidean distance to find the closer points,
    and computes the error based on the closest point.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes
        expected (np.ndarray): numpy array with the expected knee points (x, y)
        s (Strategy): enum that controls the point matching (default Strategy.expected)

    Returns:
        float: the worst case MSE
    """
    # get the knee points
    knee_points = points[knees]

    error = 0.0

    if s is Strategy.knees:
        a = knee_points
        b = expected
    elif s is Strategy.expected:
        a = expected
        b = knee_points
    elif s is Strategy.best:
        if len(expected) <= len(knee_points):
            a = expected
            b = knee_points
        else:
            a = knee_points
            b = expected
    else:
        if len(expected) >= len(knee_points):
            a = expected
            b = knee_points
        else:
            a = knee_points
            b = expected
    
    for p in a:
        distances = np.linalg.norm(b-p, axis=1)
        idx = np.argmin(distances)
        error += np.sum(np.square(p-b[idx]))

    return error / (len(a)*2.0)


def rmse(points: np.ndarray, knees: np.ndarray, expected: np.ndarray, s: Strategy = Strategy.expected) -> float:
    """
    Estimates the worst case Root Mean Squared Error (RMSE) for the given
    knee and expected points.

    Suppports different size arrays, and estimates the RMSE based 
    on the worst case.
    It uses the euclidean distance to find the closer points,
    and computes the error based on the closest point.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes
        expected (np.ndarray): numpy array with the expected knee points (x, y)
        s (Strategy): enum that controls the point matching (default Strategy.expected)

    Returns:
        float: the worst case RMSE
    """
    return math.sqrt(mse(points, knees, expected, s))
