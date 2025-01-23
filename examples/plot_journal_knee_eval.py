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

from sklearn import datasets
from sklearn.cluster import KMeans

import kneeliverse.kneedle as kneedle



logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main(args):
    for k in range(args.min, args.max+1):
        logger.info(f'{k}')
        X, y = datasets.make_blobs(n_samples=10*k, centers=k, n_features=2, random_state=42)
        logger.info(f'{X}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of Knee detection in clustering operations')
    parser.add_argument('--min', type=int, help='min number of clusters', default=5)
    parser.add_argument('--max', type=int, help='max number of clusters', default=15)
    args = parser.parse_args()
    main(args)