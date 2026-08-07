
'''
The following module provides knee detection based on the AutoElbow method.

AutoElbow scores each point by a ratio of squared distances to three fixed
references, and takes the largest. It has no tuning parameters and runs in a
single pass, which makes it a useful counterweight to the threshold-driven
methods elsewhere in this library.

Reference:
    A. J. Onumanyi, D. N. Molokomme, S. J. Isaac and A. M. Abu-Mahfouz,
    "AutoElbow: An Automatic Elbow Detection Method for Estimating the Number
    of Clusters in a Dataset", Applied Sciences 12(15):7515, 2022.
    https://doi.org/10.3390/app12157515
'''

__author__ = 'Mário Antunes'
__version__ = '1.0'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'

import logging

import numpy as np

import kneeliverse.knee_ranking as kr
import kneeliverse.multi_knee as mk
from kneeliverse import utils
from kneeliverse.utils import Concavity, Direction

logger = logging.getLogger(__name__)


def normalize(points: np.ndarray) -> np.ndarray:
    """
    Rescale both axes of a curve onto [0, 1].

    AutoElbow compares distances measured along x with distances measured
    along y, so the two have to share a scale before any of it means
    anything. A degenerate axis (every value identical) is mapped to zeros
    rather than dividing by its zero range.

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        np.ndarray: the points, each axis rescaled to [0, 1]
    """
    points = np.asarray(points, dtype=float)
    out = np.empty_like(points)
    for axis in (0, 1):
        column = points[:, axis]
        span = column.max() - column.min()
        out[:, axis] = np.zeros_like(column) if span == 0 else (column - column.min()) / span
    return out


def enforce_monotonicity(y: np.ndarray, decreasing: bool) -> np.ndarray:
    """
    Smooth away the wobbles that run against the curve's overall trend.

    AutoElbow assumes a curve that only ever falls (or only ever rises); a
    local reversal puts a spurious bulge in the distance ratio. Any point
    that moves the wrong way relative to its predecessor is replaced by its
    successor, per equations (8) and (9) of the paper.

    Args:
        y (np.ndarray): the ordinate values, already normalized
        decreasing (bool): True if the curve should only fall

    Returns:
        np.ndarray: a copy of `y` with reversals removed
    """
    y = np.array(y, dtype=float)
    for k in range(1, len(y) - 1):
        reversed_here = y[k] > y[k - 1] if decreasing else y[k] < y[k - 1]
        if reversed_here:
            y[k] = y[k + 1]
    return y


def _references(direction: Direction, concavity: Concavity) -> tuple:
    """
    The three reference points AutoElbow measures against, for each of the
    four curve shapes (Figure 2 of the paper).

    Returns (O, Q, r_y): the near corner, the far corner, and the ordinate of
    the per-point reference R = (x_k, r_y).
    """
    convex = concavity is Concavity.Counterclockwise
    if convex and direction is Direction.Decreasing:      # left elbow
        return (0.0, 0.0), (1.0, 1.0), 0.0
    if convex and direction is Direction.Increasing:      # right elbow
        return (1.0, 0.0), (0.0, 1.0), 0.0
    if direction is Direction.Increasing:                 # left knee
        return (0.0, 1.0), (1.0, 0.0), 1.0
    return (1.0, 1.0), (0.0, 0.0), 1.0                    # right knee


def score(points: np.ndarray) -> np.ndarray:
    """
    The AutoElbow function f, evaluated at every point of the curve.

    Each point is scored by

        f = b / (a + c)

    where a is its squared distance to the near corner O, b its squared
    distance to the far corner Q, and c its squared distance to R - the point
    directly above or below it on the axis. A point deep in either straight
    run sits close to one corner and far from the other, which drives the
    ratio down; the knee is where the trade-off is most favourable, so f
    peaks there.

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        np.ndarray: f evaluated at each point, parallel to `points`
    """
    direction, concavity = utils.detect_orientation(points)

    normalized = normalize(points)
    x = normalized[:, 0]
    # A left elbow and a right knee both fall; the other two rise.
    falling = ((concavity is Concavity.Counterclockwise) ==
               (direction is Direction.Decreasing))
    y = enforce_monotonicity(normalized[:, 1], decreasing=falling)

    (ox, oy), (qx, qy), ry = _references(direction, concavity)

    a = (x - ox)**2 + (y - oy)**2
    b = (x - qx)**2 + (y - qy)**2
    c = (y - ry)**2

    # a + c reaches 0 only at the near corner itself, where the ratio is
    # meaningless rather than infinite; scoring it 0 keeps it from winning.
    denominator = a + c
    return np.divide(b, denominator, out=np.zeros_like(b), where=denominator > 0)


def knee(points: np.ndarray) -> int:
    """
    Returns the index of the knee point based on the AutoElbow method.

    Unlike the other detectors here this one takes no threshold, no
    sensitivity and no smoothing window: the answer is a property of the
    curve alone. It also handles all four orientations, so it does not need
    to be told whether it is looking at a knee or an elbow.

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        int: the index of the knee point
    """
    if len(points) < 3:
        return 0

    # A curve with no variation has no knee. Left alone the ratio still has a
    # maximum - the denominator vanishes at the near corner and the point
    # beside it wins - so it would return an arbitrary index 1.
    y = np.asarray(points, dtype=float)[:, 1]
    if y.max() == y.min():
        return 0

    # argmax_tol, like every other selection site: points of equal score are
    # a genuine tie and must not be separated by last-bit arithmetic.
    return kr.argmax_tol(score(points))


def multi_knee(points: np.ndarray, t1: float = 0.001, t2: int = 3) -> np.ndarray:
    """
    Recursive knee point detection based on AutoElbow.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        t1 (float): coefficient of determination threshold (default 0.001)
        t2 (int): minimum number of points per segment (default 3)

    Returns:
        np.ndarray: the indexes of the knee points
    """
    return mk.multi_knee(knee, points, t1, t2)
