#!/usr/bin/env python3

"""
Score how sharp a corner is, using `postprocessing.triangle_area`.

The area of the triangle formed by a point and its two neighbours is a cheap,
local measure of how hard the curve bends there: collinear points enclose
nothing, a sharp corner encloses a lot. On evenly spaced x it is exactly half
the absolute second difference,

    triangle_area(p[k-1], p[k], p[k+1]) == 0.5 * |y[k-1] - 2*y[k] + y[k+1]|

so it is a discrete curvature, arrived at geometrically rather than by
differentiating twice. Because it is an absolute area it does not care which
way round the three points are listed, and it is never negative - which is
what lets you rank by it directly.

Usage:
    python -m examples.triangle_area_corners
    python -m examples.triangle_area_corners -i traces/w0.csv -n 5
"""

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'

import argparse
import logging

import numpy as np

import kneeliverse.postprocessing as pp
from kneeliverse import curvature, lmethod

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def corner_strength(points: np.ndarray) -> np.ndarray:
    """Triangle area at every interior point; the endpoints score 0."""
    strength = np.zeros(len(points))
    for i in range(1, len(points) - 1):
        strength[i] = pp.triangle_area(points[[i - 1, i, i + 1]])
    return strength


def two_corner_curve() -> np.ndarray:
    """A hard bend at x=6 and a gentler one at x=18."""
    y = np.concatenate([
        np.linspace(1.00, 0.40, 7),    # steep
        np.linspace(0.40, 0.30, 12)[1:],   # shallow
        np.full(12, 0.30),             # flat
    ])
    return np.column_stack((np.arange(len(y), dtype=float), y))


def main() -> None:
    parser = argparse.ArgumentParser(description='Rank corners by triangle area')
    parser.add_argument('-i', type=str, default=None, help='input CSV (x,y per line)')
    parser.add_argument('-n', type=int, default=3, help='how many corners to report')
    args = parser.parse_args()

    points = np.genfromtxt(args.i, delimiter=',')[:, :2] if args.i else two_corner_curve()
    strength = corner_strength(points)

    logger.info(f'curve: {len(points)} points')
    logger.info(f'\ntop {args.n} corners by triangle area:')
    logger.info(f'{"x":>8}{"y":>10}{"area":>12}')
    for i in np.argsort(strength)[::-1][:args.n]:
        logger.info(f'{points[i, 0]:>8.0f}{points[i, 1]:>10.3f}{strength[i]:>12.4f}')

    # The area is never negative, so ranking by it needs no absolute value at
    # the call site - that is the whole point of it being an area.
    assert np.all(strength >= 0.0)

    # On evenly spaced x it coincides with half the absolute second
    # difference; worth showing, because it says what is actually being
    # measured - curvature, not slope.
    if np.allclose(np.diff(points[:, 0]), points[1, 0] - points[0, 0]):
        y = points[:, 1]
        second = 0.5 * np.abs(y[:-2] - 2 * y[1:-1] + y[2:])
        agree = np.allclose(strength[1:-1], second)
        logger.info(f'\nmatches 0.5*|y[k-1] - 2y[k] + y[k+1]| : {agree}')

    sharpest = int(np.argmax(strength))
    logger.info(f'\nsharpest corner at x={points[sharpest, 0]:.0f}')
    logger.info(f'  curvature.knee says x={points[curvature.knee(points), 0]:.0f}')
    logger.info(f'  lmethod.knee   says x={points[lmethod.knee(points), 0]:.0f}')
    logger.info('\nThey answer different questions: triangle area finds the SHARPEST\n'
                'bend, the detectors find the knee - the best cost/benefit\n'
                'compromise. On a curve with one bend they agree; with several,\n'
                'the sharpest is not necessarily the one worth cutting at.')


if __name__ == '__main__':
    main()
