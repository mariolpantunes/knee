#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '1.0'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''


import logging
import argparse
import numpy as np

import kneeliverse.rdp as rdp


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main(args):
    # load trace points
    points = np.genfromtxt(args.i, delimiter=',')
    
    # reduce trace points to fixed size
    reduced = rdp.rdp_fixed(points, args.l)
    space_saving = round((1.0-(len(reduced)/len(points)))*100.0, 2)
    logger.info(f'{args.i} number of data points after RDP: {len(reduced)}({space_saving} %)')
    
    # store trace points
    points_reduced = points[reduced]
    np.savetxt(args.o, points_reduced, delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fixed RDP application')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-o', type=str, required=True, help='output file')
    parser.add_argument('-l', type=int, help='RDP fixed length', default=32)
    args = parser.parse_args()
    
    main(args)