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


def get_neighbourhood_fast_points(points: np.ndarray, a: int, b: int, t: float) -> tuple:
    """Get the neighbourhood (closest points) from a to b.

    The neighbourhood is defined as the longest straitgh line (defined by R2).
    This version uses a inaccurate binary search to speedup the search.

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
    return get_neighbourhood_fast(x, y, a, b, t)


def get_neighbourhood_binary(x: np.ndarray, y: np.ndarray, a:int, b:int, t=0.9) -> int:
    """
    Get the index of the point within the range [b, a] where the R2 is close to the threshold.

    This version uses a inaccurate binary search to speedup the search.

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates
        a (int): the initial point of the search
        b (int): the left limit of the search
        t (float): R2 threshold (default 0.9)

    Returns:
        int: index of the point
    """
    
    i = b
    right = a

    while abs(i-right) > 1:
        coef = lf.linear_fit(x[i:a+1], y[i:a+1])
        r2 = lf.linear_r2(x[i:a+1], y[i:a+1], coef)
        
        if r2 < t:
            i = int((i+right)/2.0)
        else:
            right = i
            i = int((b+right)/2.0)
            
        
    return i

    """if right != a:
        return right
    else:
        return i"""

def get_neighbourhood_fast(x: np.ndarray, y: np.ndarray, a: int, b: int, t: float = 0.9) -> tuple:
    """
    Get the neighbourhood (closest points) from a to b.

    The neighbourhood is defined as the longest straitgh line (defined by R2).
    This version uses a inaccurate binary search to speedup the search.

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates
        a (int): the initial point of the search
        b (int): the left limit of the search
        t (float): R2 threshold (default 0.9)

    Returns:
        tuple: (neighbourhood index, r2, slope)
    """
    # speedup when the search using an inaccurate binary search
    i = get_neighbourhood_binary(x, y, a, b, t)
    b, slope = lf.linear_fit(x[i:a+1], y[i:a+1])
    r2 = lf.linear_r2(x[i:a+1], y[i:a+1], (b, slope))
    previous_res = (i, r2, slope)
    
    # Linear search to improve accuracy
    while r2 < t and i < a:
        i += 1
        coef = lf.linear_fit(x[i:a+1], y[i:a+1])
        r2 = lf.linear_r2(x[i:a+1], y[i:a+1], coef)
        _, slope = coef
        previous_res = (i, r2, slope)
    
    return previous_res


def get_neighbourhood(x: np.ndarray, y: np.ndarray, a: int, b: int, t: float = 0.9) -> tuple:
    """
    Get the neighbourhood (closest points) from a to b.

    The neighbourhood is defined as the longest straitgh line (defined by R2).

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates
        a (int): the initial point of the search
        b (int): the left limit of the search
        t (float): R2 threshold (default 0.9)

    Returns:
        tuple: (neighbourhood index, r2, slope)
    """

    r2 = 1.0
    i = a - 1
    _, slope = lf.linear_fit(x[i:a+1], y[i:a+1])

    while r2 > t and i > b:
        #print('.')
        previous_res = (i, r2, slope)
        i -= 1
        coef = lf.linear_fit(x[i:a+1], y[i:a+1])
        r2 = lf.linear_r2(x[i:a+1], y[i:a+1], coef)
        _, slope = coef
        #print(f'{i} -> {r2}')

    if r2 > t:
        return i, r2, slope
    else:
        return previous_res

def accuracy_knee(points: np.ndarray, knees: np.ndarray, t: float = 0.9) -> tuple:
    """Compute the accuracy heuristic for a set of knees.

    The heuristic is based on the average distance of X and Y axis, the slope and the R2.
    In this version it is used the left neighbourhood of the knee.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes
        t (float): R2 threshold (default 0.9)

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
    for i in range(len(knees)):
        idx, r2, slope = get_neighbourhood_fast(x, y, knees[i], previous_knee)

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


def rmspe(points: np.ndarray, knees: np.ndarray, expected: np.ndarray, s: Strategy = Strategy.expected, eps:float = 1e-10) -> float:

    # get the knee points
    knee_points = points[knees]

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
    
    errors = []

    for p in a:
        distances = np.linalg.norm(b-p, axis=1)
        idx = np.argmin(distances)
        e = (p - b[idx]) / (p + eps)
        errors.extend(e)
    errors = np.array(errors)

    return np.sqrt(np.mean(np.square(errors)))
    

def cm(points: np.ndarray, knees: np.ndarray, expected: np.ndarray, t:float=0.02) -> np.ndarray:
    """
    Computes the Confusion Matrix based on the knees and expected points.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes
        expected (np.ndarray): numpy array with the expected knee points (x, y)
        t (float): the maximum allowed distance (in percentage)
    
    Returns:
        np.ndarray: the confusion matrix
    """

    dx = math.fabs(points[-1][0] - points[0][0])
    knees_points_x = points[knees][:, 0]
    tp = fn = fp = tn = 0
    for px, _ in expected:
        distances = np.fabs(knees_points_x - px)/dx
        idx = np.argmin(distances)
        if distances[idx] <= t:
            tp += 1
        else:
            fn += 1 
    fp = max(len(knees) - tp,0)
    tn = len(points) - (tp+fp+fn)

    return np.array([[tp, fp], [fn, tn]])


def accuracy(cm:np.ndarray) -> float:
    """
    Computes accuracy based on a Confusion Matrix.

    Args:
        cm (np.ndarray): the confusion matrix
    
    Returns:
        float: the accuracy
    """

    tp, fp = cm[0]
    fn, tn = cm[1]

    return (tp+tn)/(tp+tn+fp+fn)


def f1score(cm:np.ndarray) -> float:
    """
    Computes F1-Score based on a Confusion Matrix.

    Args:
        cm (np.ndarray): the confusion matrix
    
    Returns:
        float: the F1-Score
    """

    tp, fp = cm[0]
    fn, _ = cm[1]

    return (2.0*tp)/(2*tp+fp+fn)


def mcc(cm:np.ndarray) -> float:
    """
    Computes Matthews Correlation Coefficient (MCC) based on a Confusion Matrix.

    Args:
        cm (np.ndarray): the confusion matrix
    
    Returns:
        float: the mcc
    """
    
    tp, fp = cm[0]
    fn, tn = cm[1]

    n = tp*tn - fp*fn
    d = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    return n/d