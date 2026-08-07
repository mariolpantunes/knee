# coding: utf-8

'''
The following module provides a set of methods
used for ranking knees from multi-knees approaches.
'''

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''

import math
import enum
import logging
import numpy as np
import kneeliverse.linear_fit as lf
import kneeliverse.evaluation as ev


logger = logging.getLogger(__name__)


class ClusterRanking(enum.Enum):
    """
    Enum data type that represents the direction of the ranking within a cluster.
    """
    left = 'left'
    linear = 'linear'
    right = 'right'
    hull = 'hull'

    def __str__(self):
        return self.value


def distances(point:np.ndarray, points:np.ndarray) -> np.ndarray:
    """
    Computes the euclidean distance from a single point to a vector of points.

    Args:
        point (np.ndarray): the point
        points (np.ndarray): the vector of points
    
    Returns:
        np.ndarray: a vector with the distances from point to all the points. 
    """
    return np.sqrt(np.sum(np.power(points - point, 2), axis=1))


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


def rank_min(array: np.ndarray) -> np.ndarray:
    """
    Computes the rank of an array of values, with ties sharing the
    MINIMUM rank of their tied group.

    Unlike `rank`, which is built on `argsort` and therefore splits an
    exactly-tied group into consecutive integers in whatever order the
    underlying sort implementation happens to produce, this gives every
    value in a tied group the same (lowest) rank - the "competition
    ranking" convention (equivalent to `scipy.stats.rankdata(array,
    method='min') - 1`, 0-indexed). Needed whenever downstream logic
    breaks ties deterministically by a secondary key (see
    `right_flatness_ranking`): an arbitrary rank spread within a tied
    group would otherwise swamp that tie-break.

    Args:
        array (np.ndarray): array with values

    Returns:
        np.ndarray: an array with the (0-indexed, min-tie) rank of each value
    """
    return np.searchsorted(np.sort(array), array, side='left')


def rank_min_tol(array: np.ndarray, rtol: float = 1e-9, atol: float = 0.0) -> np.ndarray:
    """
    Like `rank_min`, but values that are equal to within a RELATIVE
    tolerance share a rank, instead of only bit-identical ones.

    `rank_min` ties only on exact equality, which is the wrong test for a
    value that came out of a floating-point computation. Two quantities that
    are mathematically identical - say the slope of two sub-ranges of the
    same straight line - routinely differ in the last bit or two, and
    `rank_min` then hands them a full integer rank spread built entirely out
    of that noise. Any downstream tie-break by a secondary key is swamped,
    and *which* value ends up first depends on the platform's libm and
    compiler rather than on the data. Ranking on a tolerance instead makes
    the result depend only on differences that are actually meaningful.

    Two values are grouped when they differ by no more than
    `rtol * max(|a|, |b|) + atol`. Note this is applied between ADJACENT
    values in sorted order, so grouping is transitive: a chain of values
    each within tolerance of the next collapses into one group even if its
    two ends are far apart. That is the intended reading of "no meaningful
    difference anywhere along this run", but it does mean `rtol` should stay
    small relative to the differences you care about.

    Args:
        array (np.ndarray): array with values
        rtol (float): relative tolerance for grouping (default 1e-9)
        atol (float): absolute tolerance, added to the relative term - use
            for arrays whose values legitimately reach 0.0 (default 0.0)

    Returns:
        np.ndarray: an array with the (0-indexed, tolerant-tie) rank of each
        value
    """
    array = np.asarray(array, dtype=float)
    if array.size == 0:
        return np.zeros(0, dtype=int)

    order = np.argsort(array, kind='stable')
    ordered = array[order]

    scale = np.maximum(np.abs(ordered[1:]), np.abs(ordered[:-1]))
    starts_group = np.concatenate(
        ([True], np.diff(ordered) > rtol * scale + atol))

    # Every member of a group takes the sorted position of the group's first
    # member - the same "lowest rank of the tied group" convention rank_min
    # uses, just with a tolerant notion of "tied".
    group_id = np.cumsum(starts_group) - 1
    first_position = np.flatnonzero(starts_group)

    ranks = np.empty(array.size, dtype=int)
    ranks[order] = first_position[group_id]
    return ranks


def slope_ranking(points: np.ndarray, knees: np.ndarray, t: float = 0.8) -> np.ndarray:
    """
    Computes the rank of a set of knees in a curve.

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


def smooth_ranking(points: np.ndarray, knees: np.ndarray, t: ClusterRanking) -> np.ndarray:
    """
    Computes the rank for a cluster of knees in a curve.

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

    x = points[:, 0]
    y = points[:, 1]

    fit = []
    weights = []
    
    j = knees[0]
    peak = np.max(y[knees])

    # TODO: find a better approximation, for example SMAPE
    for i in range(0, len(knees)):
        # R2 score
        r2 = 0
        if t is ClusterRanking.linear:
            r2_left = lf.r2(x[j:knees[i]+1], y[j:knees[i]+1])
            r2_right = lf.r2(x[knees[i]:knees[-1]], y[knees[i]:knees[-1]])
            r2 = (r2_left + r2_right) / 2.0
        elif t is ClusterRanking.left:
            r2 = lf.r2(x[j:knees[i]+1], y[j:knees[i]+1])
        else:
            r2 = lf.r2(x[knees[i]:knees[-1]], y[knees[i]:knees[-1]])
        fit.append(r2)

        # height of the segment
        d = math.fabs(peak - y[knees[i]])
        weights.append(d)

    #weights.append(0)
    weights = np.array(weights)
    #fit.append(0)
    fit = np.array(fit)

    #max_weights = np.max(weights)
    # if max_weights != 0:
    #    weights = weights / max_weights

    sum_weights = np.sum(weights)
    if sum_weights != 0:
        weights = weights / sum_weights

    #logger.info(f'Fit & Weights {fit} / {weights}')

    rankings = fit * weights

    #logger.info(f'Smooth Ranking {rankings}')

    return rankings


def right_flatness_ranking(points: np.ndarray, knees: np.ndarray, basis: str = 'left_ratio',
                            flatness_weight: float = 0.7, floor: float = 1e-9,
                            ratio_rtol: float = 1e-9) -> np.ndarray:
    """
    Computes the rank of a set of knees in a curve, preferring the SMALLEST
    knee whose right remainder (from the knee to the end of the curve) is
    closest to flat.

    This is the opposite lens from `slope_ranking`, which scores the
    steepness of the curve's approach INTO a knee (backward-looking, so a
    single noisy pre-knee step can inflate its score). This looks forward
    instead: a knee whose entire remainder has already flattened out has
    captured essentially all the extractable information, so further knees
    would just chase noise.

    Each knee's right-remainder slope (endpoint-to-endpoint, via
    `linear_fit` - the same 2-point approximation `slope_ranking` itself
    uses) is compared against a reference slope selected by `basis`:
      - 'left_ratio' (default): the LEFT remainder's own slope up to the
        knee - a regime-change read ("how much did the decline rate drop
        right here").
      - 'overall_ratio': the whole curve's own endpoint-to-endpoint slope,
        one global reference instead of a per-knee one.

    Combined via RANK (not the raw ratio value) with the leftmost
    preference, weighted by `flatness_weight` (0.0 = pure leftmost, 1.0 =
    pure flattest) - ranking removes the raw ratio's scale sensitivity
    (dominated by numerical noise at very large knees, where both slopes
    are near zero, or trivially tiny at very small ones, where the left
    slope is huge). Ties are broken deterministically toward the smaller
    (leftmost) knee via `rank_min_tol` plus a small leftmost nudge, so
    several equally-flat knees don't get an arbitrary winner.

    The flatness axis is ranked with a TOLERANCE (`rank_min_tol`), not with
    exact equality. Knees whose remainders are equally flat rarely produce
    bit-identical ratios: on a purely linear decline, for instance, every
    knee's ratio is 1.0 to within ~1e-15, and ranking those exactly gives a
    full integer rank spread made of nothing but floating-point noise - far
    too large for the 1e-6 leftmost nudge below to overcome, so the winner
    ends up decided by the platform's arithmetic instead of by the curve.
    `ratio_rtol` sets what counts as "no meaningful difference in flatness".

    CAVEAT: evaluated against real storage-traffic (k, cost) tradeoff
    curves from a downstream project (7 real traces, human-validated
    ground truth) - best configuration found was 1/7 exact matches,
    clearly behind that project's chosen k-selection strategy. Included
    here as a legitimate, tested alternative ranking lens, not a proven
    winner - read this caveat before assuming it is competitive by default.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        knees (np.ndarray): knees indexes
        basis (str): 'left_ratio' or 'overall_ratio' (default 'left_ratio')
        flatness_weight (float): weight of the flatness rank vs. the
            leftmost rank, in [0, 1] (default 0.7)
        floor (float): minimum reference slope, to avoid division by zero
            (default 1e-9)
        ratio_rtol (float): relative tolerance below which two knees count
            as equally flat and share a rank, leaving the leftmost nudge to
            decide between them (default 1e-9)

    Returns:
        np.ndarray: an array with the ranks of each value
    """
    x = points[:, 0]
    y = points[:, 1]

    def _slope(i: int, j: int) -> float:
        _, m = lf.linear_fit(x[i:j + 1], y[i:j + 1])
        return math.fabs(m)

    right_slopes = np.array([_slope(k, len(points) - 1) for k in knees])
    if basis == 'overall_ratio':
        overall = _slope(0, len(points) - 1)
        reference = np.full_like(right_slopes, max(overall, floor))
    else:
        reference = np.array([max(_slope(0, k), floor) for k in knees])
    ratio = right_slopes / reference

    # Tolerant on the flatness axis (a computed float, noisy in its last
    # bits); exact on the leftmost axis (knee indices, exact integers).
    flatness_rank = rank_min_tol(ratio, rtol=ratio_rtol)
    leftmost_rank = rank_min(knees)
    combined = flatness_weight * flatness_rank + (1 - flatness_weight) * leftmost_rank
    # Deterministic tie-break toward the smaller (leftmost) knee.
    combined = combined + 1e-6 * leftmost_rank

    return -combined
