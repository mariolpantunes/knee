#!/usr/bin/env python3

"""
Measure how well a simplified curve still represents the original, and
optionally render the result.

Runs each RDP variant over the same curve and reports, per variant, how many
points survived and what that reduction costs under every available metric -
using `evaluation.compute_global_segment_cost`, which returns the global cost
together with the per-segment residuals behind it. With `-p` it also writes
one frame per variant through `rdp.plot_frame`.

Usage:
    python -m examples.rdp_cost_and_frames                     # synthetic curve
    python -m examples.rdp_cost_and_frames -i traces/w0.csv    # a real trace
    python -m examples.rdp_cost_and_frames -p -o ./frames      # render frames
"""

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'

import argparse
import logging

import numpy as np

from kneeliverse import evaluation, metrics, rdp

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def synthetic_curve(n: int = 200) -> np.ndarray:
    """A miss-ratio-style curve: decaying, with a knee, bounded away from zero.

    The floor matters. `smape` and `rpd` both divide by the magnitude of the
    values, so on a curve that decays toward 0 the tail dominates the global
    cost no matter how well it is fitted - and `grdp`, which stops on that
    global cost, is then forced to keep splitting the tail and ends up
    retaining MORE points than `rdp` rather than fewer. Real traces in
    `traces/` do not have that shape (`usr0` bottoms out at 0.09, `web0` at
    0.33), so neither should the stand-in for them.
    """
    x = np.arange(n, dtype=float)
    return np.column_stack((x, 0.1 + 0.9 * np.exp(-x / (n / 8.0))))


def main() -> None:
    parser = argparse.ArgumentParser(description='RDP reduction cost report')
    parser.add_argument('-i', type=str, default=None, help='input CSV (x,y per line)')
    parser.add_argument('-t', type=float, default=0.01, help='RDP threshold (default 0.01)')
    parser.add_argument('-p', action='store_true', help='render one frame per variant')
    parser.add_argument('-o', type=str, default='./img', help='frame directory (default ./img)')
    args = parser.parse_args()

    points = np.genfromtxt(args.i, delimiter=',')[:, :2] if args.i else synthetic_curve()
    logger.info(f'curve: {len(points)} points')

    variants = {
        'rdp': lambda p: rdp.rdp(p, t=args.t),
        'grdp': lambda p: rdp.grdp(p, t=args.t),
        'rdp_fixed': lambda p: rdp.rdp_fixed(p, length=min(20, len(p))),
        'mp_grdp': lambda p: rdp.mp_grdp(p, t=args.t, min_points=min(20, len(p))),
    }

    header = f'{"variant":<12}{"kept":>6}{"ratio":>8}  ' + ''.join(f'{m.value:>12}' for m in metrics.Metrics)
    logger.info(header)
    logger.info('-' * len(header))

    for name, reduce in variants.items():
        reduced, _ = reduce(points)
        costs = [evaluation.compute_global_segment_cost(points, reduced, m)[0]
                 for m in metrics.Metrics]
        row = f'{name:<12}{len(reduced):>6}{len(reduced) / len(points):>8.1%}  '
        logger.info(row + ''.join(f'{c:>12.3e}' for c in costs))

        if args.p:
            path = rdp.plot_frame(points, reduced, name, directory=args.o)
            logger.info(f'{"":<12}wrote {path}')

    # The per-segment residuals show WHERE a reduction is paying its cost,
    # which the single global number cannot.
    reduced, _ = rdp.grdp(points, t=args.t)
    _, segments = evaluation.compute_global_segment_cost(points, reduced)
    worst = int(np.argmax(segments))
    logger.info(f'\ngrdp: worst-fitting segment is {worst} of {len(segments)}, '
                f'spanning x={points[reduced[worst], 0]:.0f}..{points[reduced[worst + 1], 0]:.0f} '
                f'(residual {segments[worst]:.3e})')


if __name__ == '__main__':
    main()
