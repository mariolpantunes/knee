# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import os
import csv
import argparse
import numpy as np

from rdp import rdp
from kneedle import auto_knee
import matplotlib.pyplot as plt


def main(args):
    points = np.genfromtxt(args.i, delimiter=',')
    print(points)

    points_reduced = rdp(points, args.r)
    print(points)
    #double space_saving = MathUtils.round((1.0-(points_rdp.size()/(double)points.size()))*100.0, 2);
    space_saving = round((1.0-(len(points_reduced)/len(points)))*100.0, 2)
    print('Number of data points after RDP: {}({}%)'.format(len(points_reduced), space_saving));

    values = auto_knee(points_reduced, debug=True)

    print(values)

    fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(3,2)
    
    
    xpoints = np.transpose(points)[0]
    ypoints = np.transpose(points)[1]
    ax0.plot(xpoints, ypoints)
    ax0.set_yticklabels([])
    ax0.set_xticklabels([])
    ax0.text(.5,.9,'Original',horizontalalignment='center', transform=ax0.transAxes)

    xpoints_reduced = np.transpose(points_reduced)[0]
    ypoints_reduced = np.transpose(points_reduced)[1]
    ax1.plot(xpoints_reduced, ypoints_reduced)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.text(.5,.9,'Reduced',horizontalalignment='center', transform=ax1.transAxes)

    xds = np.transpose(values['Ds'])[0]
    yds = np.transpose(values['Ds'])[1]
    ax2.plot(xds, yds)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.text(.5,.9,'Smooth',horizontalalignment='center', transform=ax2.transAxes)

    xdn = np.transpose(values['Dn'])[0]
    ydn = np.transpose(values['Dn'])[1]
    ax3.plot(xdn, ydn)
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax3.text(.5,.9,'Normalized',horizontalalignment='center', transform=ax3.transAxes)

    xdd = np.transpose(values['Dd'])[0]
    ydd = np.transpose(values['Dd'])[1]
    ax4.plot(xdd, ydd)
    ax4.set_yticklabels([])
    ax4.set_xticklabels([])
    ax4.text(.5,.9,'Differences',horizontalalignment='center', transform=ax4.transAxes)

    xpoints_reduced = np.transpose(points_reduced)[0]
    ypoints_reduced = np.transpose(points_reduced)[1]
    ax5.plot(xpoints_reduced, ypoints_reduced)
    ax5.set_yticklabels([])
    ax5.set_xticklabels([])
    ax5.text(.5,.9,'Knees',horizontalalignment='center', transform=ax5.transAxes)

    for i in values['knees']:
        ax5.axvline(xpoints_reduced[i], color='r')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(0, 0)
    filename = os.path.splitext(args.i)[0]+'.pdf'
    plt.savefig(filename, transparent = True, bbox_inches = 'tight', pad_inches = 0, dpi = 300)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knee testing app')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-r', type=float, help='R2', default=0.95)
    #parser.add_argument('-o', type=str, help='output file')
    args = parser.parse_args()
    main(args)