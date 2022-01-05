# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import os
import re
import enum
import logging
import argparse
import numpy as np


class Trace(enum.Enum):
    lru = 'lru'
    arc = 'arc'
    all = 'all'

    def __str__(self):
        return self.value


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main(args):
    if args.t is Trace.all:
        files = [f for f in os.listdir(args.p) if re.match(r'w[0-9]*-(lru|arc)_reduced\.csv', f)]
    elif args.t is Trace.arc:
        files = [f for f in os.listdir(args.p) if re.match(r'w[0-9]*-arc_reduced\.csv', f)]
    else:
        files = [f for f in os.listdir(args.p) if re.match(r'w[0-9]*-lru_reduced\.csv', f)]
    
    logger.info(f'{files}')

    # Create a matrix
    distances = np.zeros((len(files), len(files)))

    # Fill the matrix with the values
    for i in range (len(files)-1):
        points_i = np.genfromtxt(f'{args.p}/{files[i]}', delimiter=',')
        y_i = points_i[:, 1]
        norm = np.linalg.norm(y_i)
        y_i /= norm 
        for j in range (i+1,len(files)):
            points_j = np.genfromtxt(f'{args.p}/{files[j]}', delimiter=',')
            y_j = points_j[:, 1]
            norm = np.linalg.norm(y_j)
            y_j /= norm 
            # rmse
            distance = np.linalg.norm(y_i - y_j) / np.sqrt(len(y_i))
            distances[i,j] = distance
            distances[j,i] = distance

    logger.info(f'{distances}')

    # Process the distance matrix
    for i in range(len(distances)):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MRC compute distance')
    parser.add_argument('-p', type=str, help='input path', default='~/mrcs')
    parser.add_argument('-t', type=Trace, choices=list(Trace), default='lru')
    args = parser.parse_args()
    
    main(args)