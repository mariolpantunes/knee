import argparse
import numpy as np
import logging
import matplotlib.pyplot as plt


def main(args):
    # adjust matplotlib parameters
    #plt.rcParams.update(plt.rcParamsDefault)
    #plt.rcParams['text.usetex'] = True
    #plt.rcParams['font.size'] = 18
    #plt.rcParams['font.family'] = "serif"

    # Create main container
    fig = plt.figure()

    points = np.genfromtxt(args.i, delimiter=',')
    x = points[:, 0]
    y = points[:, 1]

    # Create zoom-in plot
    ax = plt.plot(x, y)
    #plt.xlim(400, 500)
    #plt.ylim(350, 400)
    plt.xlabel('Cache Size (GB)', labelpad = 15)
    plt.ylabel('Miss Ratio', labelpad = 15)

    # Create zoom-out plot
    ax_new = fig.add_axes([0.6, 0.6, 0.2, 0.2]) # the position of zoom-out plot compare to the ratio of zoom-in plot 
    t = 0.1
    idx = int(len(points)*(1.0-t))
    print(f'{len(points)} -> {idx}')
    plt.plot(x[idx:], y[idx:])
    #plt.xlabel('Cache Size (GB)', labelpad = 15)
    #plt.ylabel('Miss Ratio', labelpad = 15)

    # Save figure with nice margin
    plt.savefig('/home/mantunes/mrcs/plots/00.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = .1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RDP test application')
    parser.add_argument('-i', type=str, required=True, help='input file')
    args = parser.parse_args()
    
    main(args)