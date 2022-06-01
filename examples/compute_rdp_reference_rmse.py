# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import os
import re
import csv
import enum
import tqdm
import signal
import pathlib
import logging
import argparse
import numpy as np


import knee.rdp as rdp
import knee.linear_fit as lf
import knee.metrics as metrics


class Trace(enum.Enum):
    lru = 'lru'
    arc = 'arc'
    all = 'all'

    def __str__(self):
        return self.value


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)





def main(args):
    ## Path for all the MRCs
    path = os.path.expanduser(args.p)
    if args.tr is Trace.all:
        files = [f for f in os.listdir(path) if re.match(r'w[0-9]*-(lru|arc)\.csv', f)]
    elif args.tr is Trace.arc:
        files = [f for f in os.listdir(path) if re.match(r'w[0-9]*-arc\.csv', f)]
    else:
        files = [f for f in os.listdir(path) if re.match(r'w[0-9]*-lru\.csv', f)]

    for f in tqdm.tqdm(files, position=0, desc='MRC', leave=False):
        points = np.genfromtxt(f'{path}{f}', delimiter=',')
        
        # the reference RMSE is computed based on the simplest line possible
        reduced = [0, len(points)-1]
        cost = compute_global_rmse(points, reduced)

        # open the corret csv file and write the result
        with open(f'/tmp/reference_rmse.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow([pathlib.Path(f).stem, cost, len(reduced)])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate all the RDPs configurations')
    parser.add_argument('-p', type=str, help='input path', default='~/mrcs/')
    parser.add_argument('-tr', type=Trace, choices=list(Trace), default='all')
    args = parser.parse_args()
    main(args)