# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
import numpy as np
import knee.convex_hull as ch
import uts.gradient as grad
import uts.thresholding as thresh


logger = logging.getLogger(__name__)


def knee(points: np.ndarray) -> int:
    pass


def multi_knee(points: np.ndarray) -> np.ndarray:
    hull = ch.graham_scan_lower(points)
    hull_points = points[hull]
    x = hull_points[:, 0]
    y = hull_points[:, 1]
    #gradient = np.absolute(grad.cfd(x, y))
    gradient = grad.cfd(x, y)
    logger.debug(f'Gradient: {gradient}')
    t = thresh.isodata(gradient)
    logger.debug(f'Threshold: {t}')
    knees = hull[gradient>=t]
    return knees
