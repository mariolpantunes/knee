#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import os
import csv
import math
import argparse
import numpy as np
import logging


from enum import Enum
import knee.zmethod as zmethod
import knee.evaluation as evaluation


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class Clustering(Enum):
    single = 'single'
    complete = 'complete'
    average = 'average'

    def __str__(self):
        return self.value


class Evaluation(Enum):
    regression = 'regression'
    classification = 'classification'

    def __str__(self):
        return self.value


def main(args):
    # define clustering methods
    # cmethod = {Clustering.single: clustering.single_linkage, Clustering.complete: clustering.complete_linkage, Clustering.average: clustering.average_linkage}
    # get the expected file from the input file
    dirname = os.path.dirname(args.i)
    filename = os.path.splitext(os.path.basename(args.i))[0]
    expected_file = os.path.join(os.path.normpath(dirname), f'{filename}_expected.csv')

    expected = None

    if os.path.exists(expected_file):
        with open(expected_file, 'r') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            expected = list(reader)
    else:
        expected = []
    expected = np.array(expected)
    points = np.genfromtxt(args.i, delimiter=',')

    ## Knee detection code ##

    knees = zmethod.knees(points)
    knees = knees[knees>0]

    ##########################

    if args.e is Evaluation.regression:
        logger.info(f'MSE(knees)   MSE(exp)   Cost(tr)   Cost(kn) RMSPE(knees) RMPSE(exp)')
        logger.info(f'-------------------------------------------------------------------')
        if len(expected) > 0:
            error_mse = evaluation.mse(points, knees, expected, evaluation.Strategy.knees)
            error_mse_exp = evaluation.mse(points, knees, expected, evaluation.Strategy.expected)
            error_rmspe = evaluation.rmspe(points, knees, expected, evaluation.Strategy.knees)
            error_rmspe_exp = evaluation.rmspe(points, knees, expected, evaluation.Strategy.expected)
        else:
            error_mse = math.nan
            error_mse_exp = math.nan
            error_rmspe = math.nan
            error_rmspe_exp = math.nan
        _,_,_,_,cost_trace = evaluation.accuracy_trace (points, knees)
        _,_,_,_,cost_knee = evaluation.accuracy_knee (points, knees)
        logger.info(f'{error_mse:10.2E} {error_mse_exp:10.2E} {cost_trace:10.2E} {cost_knee:10.2E} {error_rmspe:12.2E} {error_rmspe_exp:10.2E}')
    else:
        logger.info(f'Accuracy F1-Score  MCC')
        logger.info(f'----------------------')
        cm = evaluation.cm(points, knees, expected)
        accuracy = evaluation.accuracy(cm)
        f1score = evaluation.f1score(cm)
        mcc = evaluation.mcc(cm)
        logger.info(f'{accuracy:8.2} {f1score:8.2} {mcc:4.2}')

    # store outpout
    if args.o:
        dirname = os.path.dirname(args.i)
        filename = os.path.splitext(os.path.basename(args.i))[0]
        output = os.path.join(os.path.normpath(dirname), f'{filename}_output.csv')

        dataset = points[knees]

        with open(output, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi Knee evaluation app')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-e', type=Evaluation, choices=list(Evaluation), help='Evaluation type', default='regression')
    parser.add_argument('-o', help='store output (debug)', action='store_true')
    args = parser.parse_args()
    
    main(args)
