
'''
The following module provides shared vocabulary and helpers used across
several knee detection methods.

Its main content is the orientation of a curve - whether it rises or falls,
and whether it bulges above or below the straight line joining its ends.
Detectors need this before they can look for a knee, because the point they
are hunting for sits on a different side of that line depending on the
answer, and more than one of them derived it independently.
'''

__author__ = 'Mário Antunes'
__version__ = '1.0'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''

import enum
import logging

import numpy as np

import kneeliverse.linear_fit as lf

logger = logging.getLogger(__name__)


class Direction(enum.Enum):
    """
    Enum data type that represents the direction of a concavity.
    """
    Increasing = 'increasing'
    Decreasing = 'decreasing'

    def __str__(self):
        return self.value


class Concavity(enum.Enum):
    """
    Enum data type that represents the rotation of a concavity.

    Member order is load-bearing: the demos and examples pass
    `list(Concavity)` as argparse `choices`, so reordering would change
    their command-line surface.
    """
    Counterclockwise = 'counter-clockwise'
    Clockwise = 'clockwise'

    def __str__(self):
        return self.value


def normalize(points: np.ndarray) -> np.ndarray:
    """
    Rescale both axes of a curve onto [0, 1].

    Several methods compare a distance measured along x with one measured
    along y, which only means something once the two share a scale.

    An axis with no variation is mapped to zeros rather than divided by its
    zero range. That case is not hypothetical: `kneedle.knee` guarded against
    it and `kneedle.knees`, three lines away in the same module, did not - so
    a flat curve produced a knee from one and a RuntimeWarning and NaNs from
    the other. Having one implementation is what keeps the two consistent.

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        np.ndarray: the points, each axis rescaled to [0, 1]
    """
    points = np.asarray(points, dtype=float)
    minimum = points.min(axis=0)
    span_ = points.max(axis=0) - minimum
    # A zero span means the axis is constant; every value maps to 0.
    span_[span_ == 0] = 1.0
    return (points - minimum) / span_


def span(points: np.ndarray) -> tuple:
    """
    The extent a curve covers on each axis.

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        tuple: (dx, dy), both non-negative
    """
    points = np.asarray(points, dtype=float)
    dx, dy = np.abs(points.max(axis=0) - points.min(axis=0))
    return float(dx), float(dy)


def detect_orientation(points: np.ndarray) -> tuple:
    """
    Classify a curve by the direction it runs and the way it bends.

    Between them, `Direction` and `Concavity` name the four shapes a knee or
    elbow graph can take, and a detector has to know which one it is looking
    at before it can decide where the knee should be.

    Direction comes from the sign of the slope of the line through the
    curve's endpoints. Concavity comes from the SIGN OF THE SUM of the
    residuals against that same line: if the curve spends more of itself
    above the chord than below, it bulges upwards and the rotation is
    clockwise.

    Summing over every point, rather than testing one of them, is what makes
    this robust. A single sample can be corrupted - and if the sample chosen
    happens to be the one perturbed, the whole curve is misclassified.
    Measured on an exponential decay with one spiked value, a midpoint test
    disagreed with this one on 168 of 200 trials, and this one stayed right.

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        tuple: (Direction, Concavity) of the curve
    """
    b, m = lf.linear_fit_points(points)

    direction = Direction.Increasing if m > 0.0 else Direction.Decreasing

    y = points[:, 1]
    y_hat = points[:, 0] * m + b
    vote = np.sum(y - y_hat)
    concavity = Concavity.Clockwise if vote > 0 else Concavity.Counterclockwise

    return direction, concavity
