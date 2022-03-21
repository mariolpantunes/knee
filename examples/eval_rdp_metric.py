# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import os
import re
import csv
import enum
import logging
import argparse
import numpy as np

import knee.rdp as rdp
import knee.linear_fit as lf
import knee.zmethod as zmethod
import knee.evaluation as evaluation


from tqdm import tqdm


class Trace(enum.Enum):
    lru = 'lru'
    arc = 'arc'
    all = 'all'

    def __str__(self):
        return self.value


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)



def main(args):

    path = os.path.expanduser(args.p)

    if args.tr is Trace.all:
        files = [f for f in os.listdir(path) if re.match(r'w[0-9]*-(lru|arc)\.csv', f)]
    elif args.tr is Trace.arc:
        files = [f for f in os.listdir(path) if re.match(r'w[0-9]*-arc\.csv', f)]
    else:
        files = [f for f in os.listdir(path) if re.match(r'w[0-9]*-lru\.csv', f)]
    
    scores = []

    for i in tqdm(range(len(files))):
        points = np.genfromtxt(f'{path}{files[i]}', delimiter=',')
        # open expected file
        dirname = os.path.dirname(f'{path}{files[i]}')
        filename = os.path.splitext(os.path.basename(files[i]))[0]
        expected_file = os.path.join(os.path.normpath(dirname), f'{filename}_expected.csv')
        expected = None
        if os.path.exists(expected_file):
            with open(expected_file, 'r') as f:
                reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
                expected = list(reader)
        else:
            expected = []
        expected = np.array(expected)

        # get original x_max and y_ranges
        x_max = [max(x) for x in zip(*points)][0]
        y_range = [[max(y),min(y)] for y in zip(*points)][1]

        # run rdp
        reduced, removed = rdp.rdp(points, t=args.r, cost=args.c, distance=args.d)
        points_reduced = points[reduced]

        ## Knee detection code ##
        knees = zmethod.knees(points_reduced, dx=args.x, dy=args.y, dz=args.z, x_max=x_max, y_range=y_range)
        knees = knees[knees>0]
        knees = rdp.mapping(knees, reduced, removed)
        if len(knees) > 0:
            cm = evaluation.cm(points, knees, expected)
            mcc = evaluation.mcc(cm)
        else:
            mcc = 0.0
        scores.append(mcc)

    # output the results
    dirname = os.path.expanduser(args.p)
    output = os.path.join(os.path.normpath(dirname), f'eval_rdp_metric_output.csv')

    with open(output, 'w') as f:
        writer = csv.writer(f)
        for s in scores:
            writer.writerow([s])        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find worst MRC trace')
    parser.add_argument('-p', type=str, help='input path', default='~/mrcs/')
    parser.add_argument('-tr', type=Trace, choices=list(Trace), default='all')
    parser.add_argument('-c', type=lf.Linear_Metrics, choices=list(lf.Linear_Metrics), default='rpd')
    parser.add_argument('-d', type=rdp.RDP_Distance, choices=list(rdp.RDP_Distance), default='shortest')
    parser.add_argument('-r', type=float, help='RDP reconstruction threshold', default=0.01)
    parser.add_argument('-x', type=float, help='Parameter dx', default=0.01)
    parser.add_argument('-y', type=float, help='Parameter dy', default=0.01)
    parser.add_argument('-z', type=float, help='Parameter dz', default=0.5)
    
    args = parser.parse_args()
    
    main(args)