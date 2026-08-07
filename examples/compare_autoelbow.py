#!/usr/bin/env python3

"""
Compare AutoElbow against the other single-knee detectors in the library.

Three views, in the order that matters:

1. Accuracy on curves whose corner is known by construction.
2. Stability as the tail grows. An elbow graph is usually produced by
   sweeping k = 1..K, and K is chosen before anyone knows where the knee
   is, so a detector that shifts its answer when you extend a curve it has
   already seen is reporting a property of your sweep rather than of your
   data. This is the paper's headline claim for AutoElbow - and on these
   curves it does not reproduce as a win, which the output says plainly.
3. Parameter sensitivity. Every other detector here has a knob; AutoElbow
   has none. The knob is not free - it has to be set by someone, usually
   without ground truth to set it against. This is where it does win.

The script is not an argument for AutoElbow. It is a way of seeing how far
apart six reasonable detectors land on the same curve, which on real traces
is further than one might hope.

Usage:
    python -m examples.compare_autoelbow
    python -m examples.compare_autoelbow -i traces/usr0.csv
"""

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'

import argparse
import glob
import logging
import os

import numpy as np

from kneeliverse import autoelbow, curvature, dfdt, kneedle, lmethod, menger, rdp

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DETECTORS = {
    'autoelbow': autoelbow.knee,
    'curvature': curvature.knee,
    'dfdt': dfdt.knee,
    'kneedle': kneedle.knee,
    'lmethod': lmethod.knee,
    'menger': menger.knee,
}


def detect(name, points) -> int:
    """Every detector shares `knee(points) -> int`; kneedle may return None.

    A detector that cannot answer for a given curve reports -1 rather than
    stopping the comparison - the point of the table is to see where they
    diverge, and a refusal is itself a result worth showing.
    """
    try:
        idx = DETECTORS[name](points)
    except (ValueError, IndexError, ZeroDivisionError) as e:
        logger.debug(f'{name} failed on a {len(points)}-point curve: {e}')
        return -1
    return -1 if idx is None else int(idx)


def corner_curve(corner: int, n: int = 30) -> np.ndarray:
    """Straight descent to `corner`, flat after it."""
    y = np.concatenate([np.linspace(1.0, 0.3, corner + 1), np.full(n - corner - 1, 0.3)])
    return np.column_stack((np.arange(len(y), dtype=float), y))


def decay_curve(n: int, tau: float = 6.0) -> np.ndarray:
    """A smooth decay bounded away from zero - the shape of a real MRC."""
    x = np.arange(n, dtype=float)
    return np.column_stack((x, 0.1 + 0.9 * np.exp(-x / tau)))


def accuracy_table() -> None:
    logger.info('\n1. ACCURACY  (corner known by construction; |error| in indexes)')
    corners = [5, 8, 11, 14, 17, 20]
    header = f'{"detector":<12}' + ''.join(f'{c:>6}' for c in corners) + f'{"mean |err|":>12}'
    logger.info(header)
    logger.info('-' * len(header))
    for name in DETECTORS:
        errors = [abs(detect(name, corner_curve(c)) - c) for c in corners]
        row = f'{name:<12}' + ''.join(f'{e:>6}' for e in errors)
        logger.info(row + f'{np.mean(errors):>12.2f}')


def stability_table() -> None:
    lengths = [20, 30, 40, 60, 80, 120]

    logger.info('\n2a. STABILITY, SHARP CORNER  (corner fixed at 10; only the tail grows)')
    header = f'{"detector":<12}' + ''.join(f'{n:>6}' for n in lengths) + f'{"spread":>9}'
    logger.info(header)
    logger.info('-' * len(header))
    for name in DETECTORS:
        picks = [detect(name, corner_curve(10, n=n)) for n in lengths]
        valid = [p for p in picks if p >= 0]
        spread = (max(valid) - min(valid)) if valid else -1
        logger.info(f'{name:<12}' + ''.join(f'{p:>6}' for p in picks) + f'{spread:>9}')

    logger.info('\n2b. STABILITY, SMOOTH CURVE  (exponential decay; no corner to anchor on)')
    logger.info(header)
    logger.info('-' * len(header))
    for name in DETECTORS:
        picks = [detect(name, decay_curve(n)) for n in lengths]
        valid = [p for p in picks if p >= 0]
        spread = (max(valid) - min(valid)) if valid else -1
        logger.info(f'{name:<12}' + ''.join(f'{p:>6}' for p in picks) + f'{spread:>9}')
    logger.info('   spread = max - min. Read it with care: curvature and menger score 0\n'
                '   by always answering 1, which is where a smooth exponential really does\n'
                '   bend hardest. That is degeneracy holding still, not robustness.')


def sensitivity_table() -> None:
    logger.info('\n3. PARAMETER SENSITIVITY  (same curve, detector knob swept)')
    points = decay_curve(60)
    logger.info(f'{"detector":<24}{"parameter":<14}{"answers":<22}{"distinct":>9}')
    logger.info('-' * 69)

    picks = [int(kneedle.knee(points, t=t)) for t in (0.5, 1.0, 2.0, 5.0)
             if kneedle.knee(points, t=t) is not None]
    logger.info(f'{"kneedle":<24}{"t":<14}{picks!s:<22}{len(set(picks)):>9}')

    picks = [int(lmethod.knee(points, fit=f)) for f in lmethod.Fit]
    logger.info(f'{"lmethod":<24}{"fit":<14}{picks!s:<22}{len(set(picks)):>9}')

    picks = [int(autoelbow.knee(points))]
    logger.info(f'{"autoelbow":<24}{"-":<14}{picks!s:<22}{len(set(picks)):>9}')
    logger.info('   AutoElbow has no knob to sweep: one curve, one answer.')


def traces_table(paths) -> None:
    logger.info('\n4. REAL TRACES  (RDP-reduced, index into the reduced curve)')
    header = f'{"trace":<24}' + ''.join(f'{n[:9]:>11}' for n in DETECTORS)
    logger.info(header)
    logger.info('-' * len(header))
    for path in paths:
        points = np.genfromtxt(path, delimiter=',')[:, :2].astype(float)
        reduced, _ = rdp.grdp(points, t=0.005)
        used = points[reduced]
        row = f'{os.path.basename(path)[:23]:<24}'
        row += ''.join(f'{detect(n, used):>11}' for n in DETECTORS)
        logger.info(row + f'   ({len(used)} pts)')


def main() -> None:
    parser = argparse.ArgumentParser(description='AutoElbow vs the other detectors')
    parser.add_argument('-i', type=str, default=None, help='a single trace to use')
    args = parser.parse_args()

    accuracy_table()
    stability_table()
    sensitivity_table()

    paths = [args.i] if args.i else sorted(glob.glob('traces/*.csv'))
    if paths:
        traces_table(paths)

    logger.info('\nWhat this actually shows, on these curves:\n'
                '  - On sharp corners AutoElbow is slightly WORSE than curvature, dfdt,\n'
                '    lmethod and menger, which are exact. Its monotonicity cleaning pulls\n'
                '    the estimate left on long descents.\n'
                '  - The stability claim does not reproduce as a win here. On a smooth\n'
                '    decay AutoElbow drifts with the sweep length much as kneedle and\n'
                '    dfdt do; only lmethod holds a non-degenerate answer.\n'
                '  - What is unambiguous is that it has nothing to tune. kneedle returns\n'
                '    four different knees over a plausible range of t, and someone has to\n'
                '    pick one - usually without ground truth to pick it against.\n'
                '  - On real traces the six disagree substantially. No detector here is\n'
                '    right by default; the useful move is to run several and look at the\n'
                '    spread, which is what this script is for.')


if __name__ == '__main__':
    main()
