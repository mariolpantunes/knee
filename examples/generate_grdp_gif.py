#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''


import tqdm
import logging
import imageio
import argparse
import numpy as np

import kneeliverse.rdp as rdp

import matplotlib.pyplot as plt


logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def generate_frame(points, reduced):
    fig = plt.figure(figsize=(10, 10))

    x = points[:, 0]
    y = points[:, 1]
    plt.plot(x, y)

    points_reduced = points[reduced]
    x = points_reduced[:, 0]
    y = points_reduced[:, 1]
    plt.plot(x, y, marker='o', markersize=3)

    fig.canvas.draw()
    image = np.array(fig.canvas.buffer_rgba())

    plt.close()
    return image


def main(args):
    logger.info(f'Loading {args.i} file...')
    points = np.genfromtxt(args.i, delimiter=',')

    frames = []

    for i in tqdm.tqdm(range(args.min, args.max)):
        reduced, _ = rdp.rdp_fixed(points, length=i, distance=args.dst, order=args.ord)
        frame = generate_frame(points, reduced)
        frames.append(frame)

    imageio.mimsave(args.o, frames, duration=500)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a GIF with the results from gRDP')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('--min', type=int, help='minimum number of points', default=3)
    parser.add_argument('--max', type=int, help='maximum number of points', default=20)
    parser.add_argument('--dst', type=rdp.Distance, choices=list(rdp.Distance), help='distance metric', default='shortest')
    parser.add_argument('--ord', type=rdp.Order, choices=list(rdp.Order), help='ordering metric', default='segment')
    parser.add_argument('-o', type=str, help='output file', default='out/gRDP.gif')
    args = parser.parse_args()
    main(args)