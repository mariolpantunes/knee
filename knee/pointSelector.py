import numpy as np
import logging
from math import ceil, fabs
from uts.zscore import zscore_array
import uts.gradient as grad
#from PyMimircache import Cachecow
#from scipy.misc import derivative
#from scipy.stats import zscore


logger = logging.getLogger(__name__)


def dict_to_lists(d):
    x = []
    y = []
    for k, v in d.items():
        x.append(k)
        y.append(v)
    return (np.array(x), np.array(y))


def dict_to_points(d):
    points = []
    for k, v in d.items():
        points.append([k, v])
    return np.array(points)


def points_to_dict(points):
    rv = {}
    for point in points:
        rv[point[0]] = point[1]
    return rv


def lists_to_dict(x, y):
    rv = {}
    for i in range(len(x)):
        rv[x[i]] = y[i]
    return rv


def map_index(a, b):
    sort_idx = np.argsort(a)
    out = sort_idx[np.searchsorted(a, b, sorter=sort_idx)]
    return out

    # index =
    #sorted_x = x[index]
    #sorted_index = np.searchsorted(sorted_x, y)


def get_knees_points(points):
    x = points[:, 0]
    y = points[:, 1]
    return get_knees(x, y)


def get_knees(x, y):
    mrc = lists_to_dict(x, y)
    rv = getPoints(mrc)
    # convert x points into indexes:
    return map_index(x, np.array(rv))


def getPoints(mrc, x_dist=0.05, y_dist=0.05, delta_z=0.05, plot=False):
    '''
    use our outlier method to find interesting points in an MRC
    @mrc: MRC dict of some trace
    @x_dist: % of max cache size between points
    @y_dist: % of max - min miss ratio between points
    @plot: set True if you want to return data useful for plotting
    '''

    uniq_reqs = len(mrc)

    if uniq_reqs < 3:
        print('pointSelector: < 3 unique requests in workload')
        return []

    if min(mrc.values()) == 1:
        logger.warning(
            'pointSelector: workload completely random (dont bother caching)')
        return []

    # get absolute x and y distances
    x_width = max(1, int(uniq_reqs * x_dist))
    y_height = (max(mrc.values()) - min(mrc.values())) * y_dist

    # get 2nd derivative yd2 using central difference formula
    #x_range = np.linspace(1, uniq_reqs, uniq_reqs, dtype=int)
    # def mrc_wrapper(x):
    # return np.array([mrc[int(i)] for i in x])
    #yd2_temp = derivative(mrc_wrapper, x_range[1:-1], dx=1.0, n=2)
    # since using central dif, first and last element can not be calculated
    #yd2 = [0] + list(yd2_temp) + [0]

    x, y = dict_to_lists(mrc)
    #yd2 = np.gradient(y, x, edge_order=2)
    yd2 = grad.csd(x, y)
    #logger.info('yd2 = %s', yd2)

    # remove negative 2nd derivatives
    #yd2 = np.array([x if x > 0 else 0 for x in yd2])
    #yd2 = [y if y > 0 else fabs(y) for y in yd2]
    yd2 = np.fabs(yd2)
    # get zscore of yd2
    #z_yd2 = zscore(yd2)
    z_yd2 = zscore_array(x, yd2)
    min_zscore = min(z_yd2)

    # optimization: create an mrc with points that have >= 0 z-score
    zero_mrc = {x[0]: [x[1], z] for x, z in zip(mrc.items(), z_yd2) if z >= 0}

    # main loop. start with outliers >= 3 z-score
    outlier_z = 3
    outlier_points = {}
    while True:

        # need to track points added each iteration for optimized outlier_dict handling
        points_added = 0
        # outlier_dict = {x:[y,z] for x,y,z in zip(list(mrc.keys()), list(mrc.values()), z_yd2) if (z >= outlier_z)
        #	and all(abs(x-i) >= x_width for i in outlier_points.keys()) and
        #	all(abs(y-j) >= y_height for j in outlier_points.values())}

        # optimization: use zero_mrc (which is much smaller) until z < 0
        if outlier_z >= 0:
            outlier_dict = {x[0]: x[1][0] for x in zero_mrc.items() if (x[1][1] >= outlier_z)
                            and all((abs(x[0]-i) >= x_width) and (abs(x[1][0]-j) >= y_height) for i, j in outlier_points.items())}
        elif outlier_z == (0 - delta_z):
            outlier_dict = {x[0]: x[1] for x, y in zip(mrc.items(), z_yd2) if (y >= outlier_z)
                            and all((abs(x[0]-i) >= x_width) and (abs(x[1]-j) >= y_height) for i, j in outlier_points.items())}

        group = []
        for k, v in sorted(outlier_dict.items()):
            if len(group) == 0:
                group += [[k, v]]
            elif (k - group[-1][0]) < x_width:
                group += [[k, v]]
            else:
                # NOTE: outlier_best is only picking from the points in the group, NOT the entire MRC window
                outlier_best = min(group, key=lambda x: x[1])
                if all(abs(outlier_best[1]-i) >= y_height for i in outlier_points.values()):
                    outlier_points[outlier_best[0]] = outlier_best[1]
                    points_added += 1
                group = [[k, v]]
        if len(group) > 0:
            outlier_best = min(group, key=lambda x: x[1])
            if all(abs(outlier_best[1]-i) >= y_height for i in outlier_points.values()):
                outlier_points[outlier_best[0]] = outlier_best[1]
                points_added += 1

        # terminating condition. there are no more candidate points
        if outlier_z <= min_zscore and points_added == 0:
            break

        outlier_z -= delta_z

    # sweep through and points to avoid picking concavity issues
    outlier_min_mr = 1.0
    outlier_keys = list(sorted(outlier_points.keys()))
    for k in outlier_keys:
        if outlier_points[k] > outlier_min_mr:
            del outlier_points[k]
        else:
            outlier_min_mr = outlier_points[k]

    # returns sorted list of cache sizes
    if not plot:
        return sorted(outlier_points.keys())
    else:
        return (outlier_points, z_yd2)
