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

import os
import re
import csv
import enum
import tqdm
import pathlib
import logging
import argparse
import numpy as np


import knee.rdp as rdp
import knee.linear_fit as lf
import knee.metrics as metrics
import knee.evaluation as evaluation


import exectime.timeit as timeit


class Trace(enum.Enum):
    lru = 'lru'
    arc = 'arc'
    all = 'all'

    def __str__(self):
        return self.value


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def compute_global_rmse(points: np.ndarray, reduced: np.ndarray):
    y, y_hat = [], []

    left = reduced[0]
    for i in range(1, len(reduced)):
        right = reduced[i]
        pt = points[left:right+1]
        
        coef = lf.linear_fit_points(pt)
        y_hat_temp = lf.linear_transform_points(pt, coef)
        
        y_hat.extend(y_hat_temp)
        y_temp = pt[:, 1]
        y.extend(y_temp)

        left = right

    # compute the cost function
    return metrics.rmse(np.array(y), np.array(y_hat))


def main(args):
    ## Path for all the MRCs
    path = os.path.expanduser(args.p)
    if args.tr is Trace.all:
        files = [f for f in os.listdir(path) if re.match(r'w[0-9]*-(lru|arc)\.csv', f)]
    elif args.tr is Trace.arc:
        files = [f for f in os.listdir(path) if re.match(r'w[0-9]*-arc\.csv', f)]
    else:
        files = [f for f in os.listdir(path) if re.match(r'w[0-9]*-lru\.csv', f)]
    
    ## RDP threshold
    rdp_threshold = [0.05, 0.01, 0.001]
    
    ## RDP Metric
    rdp_metrics = list(metrics.Metrics)

    ## RDP Order
    rdp_order = list(rdp.Order)

    for f in tqdm.tqdm(files, position=0, desc='MRC', leave=False):
        points = np.genfromtxt(f'{path}{f}', delimiter=',')
        
        for t in tqdm.tqdm(rdp_threshold, position=1, desc='Thr', leave=False):
            for c in tqdm.tqdm(rdp_metrics, position=2, desc='Cst', leave=False):
                for o in tqdm.tqdm(rdp_order, position=3, desc='Ord', leave=False):
                    
                    # convert the threhold from cost to similarity
                    if c is metrics.Metrics.r2:
                        r = 1.0 - t
                    else:
                        r = t
                    
                    ti, std, rv = timeit.timeit(args.n, rdp.grdp, points, t=r, cost=c, order=o)
                    reduced, _ = rv
                    cost = compute_global_rmse(points, reduced)
                    mip, mad = evaluation.mip(points, reduced)
                    
                    # open the corret csv file and write the result
                    with open(f'out/grdp_{t}_{c}_{o}.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                        writer.writerow([pathlib.Path(f).stem, cost, len(reduced), mip, mad, ti, std])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate all the gRDPs configurations')
    parser.add_argument('-p', type=str, help='input path', default='~/mrcs/')
    parser.add_argument('-tr', type=Trace, choices=list(Trace), help='type of traces', default='all')
    parser.add_argument('-n', type=int, help='number of repetition (for timeit)', default=5)
    args = parser.parse_args()
    main(args)