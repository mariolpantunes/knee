# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
import numpy as np
import knee.multi_knee as mk
from knee.rdp import perpendicular_distance


logger = logging.getLogger(__name__)


def get_knee(points):
    distances = perpendicular_distance(points)
    return  np.argmax(distances)


def multi_knee(points, t1=0.99, t2=2):
    return mk.multi_knee(get_knee, points, t1, t2)
