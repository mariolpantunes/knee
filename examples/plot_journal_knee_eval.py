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

import time
import logging
import argparse
import numpy as np

from sklearn import datasets
from sklearn.cluster import KMeans

import kneeliverse.dfdt as dfdt
import kneeliverse.menger as menger
import kneeliverse.lmethod as lmethod
import kneeliverse.kneedle as kneedle



logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main(args):

    # store results
    results={'menger':{'time':[], 'error':[]},
    'lmethod':{'time':[], 'error':[]},
    'dfdt':{'time':[], 'error':[]},
    'kneedle':{'time':[], 'error':[]}
    }

    for k in range(args.min, args.max+1):
        logger.debug(f'Knee {k}')
        X, y = datasets.make_blobs(n_samples=10*k, centers=k, n_features=2, random_state=42)
        points_x = np.zeros(18)
        points_y = np.zeros(18)
        for i in range(2, 20):
            kmeans = KMeans(n_clusters=i, init='k-means++', n_init='auto', random_state=42)
            kmeans.fit(X)
            points_x[i-2] = i
            points_y[i-2] = kmeans.inertia_
        points = np.stack((points_x, points_y), axis=1)
        logger.debug(f'{points}')
        for m in ['menger', 'lmethod', 'dfdt', 'kneedle']:
            if m == 'menger':
                start_time = time.process_time_ns()
                knee = menger.knee(points=points)
                end_time = time.process_time_ns()
                time_knee = end_time - start_time
            elif m == 'lmethod':
                start_time = time.process_time_ns()
                knee = lmethod.knee(points=points)
                end_knee = time.process_time_ns()
                time_init = end_time - start_time
            elif m == 'dfdt':
                start_time = time.process_time_ns()
                knee = dfdt.knee(points=points)
                end_time = time.process_time_ns()
                time_knee = end_time - start_time
            elif m == 'kneedle':
                start_time = time.process_time_ns()
                knee = kneedle.knee(points=points)
                end_time = time.process_time_ns()
                time_knee = end_time - start_time
            results[m]['time'].append(time_knee)
            results[m]['error'].append(abs(k - int(knee)))
    logger.debug(f'{results}')

    for m in ['menger', 'lmethod', 'dfdt', 'kneedle']:
        print(f'{m} | ', end='')
        for t,e in zip(results[m]['time'], results[m]['error']):
            print(f'{t:.2E} {e} |', end='')
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of Knee detection in clustering operations')
    parser.add_argument('--min', type=int, help='min number of clusters', default=5)
    parser.add_argument('--max', type=int, help='max number of clusters', default=10)
    args = parser.parse_args()
    main(args)