# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
import numpy as np
from knee.rdp import perpendicular_distance


logger = logging.getLogger(__name__)


def get_knee(points):
    distnaces = perpendicular_distance(points)
    return  np.argmax(distnaces)

