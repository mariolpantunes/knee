# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import enum
import logging
import numpy as np
from knee.linear_fit import linear_fit_points
import knee.multi_knee as mk
from uts import ema, peak_detection


logger = logging.getLogger(__name__)


class Direction(enum.Enum):
    Increasing='increasing'
    Decreasing='decreasing'

    def __str__(self):
        return self.value


class Concavity(enum.Enum):
    Counterclockwise='counter-clockwise'
    Clockwise='clockwise'

    def __str__(self):
        return self.value


def differences(points, cd, cc):
    rv = np.empty(points.shape)
    
    if cd is Direction.Decreasing and cc is Concavity.Clockwise:
        for i in range(0, len(points)):
            rv[i][0] = points[i][0]
            rv[i][1] = points[i][0] + points[i][1] # x + y
    elif cd is Direction.Decreasing and cc is Concavity.Counterclockwise:
        for i in range(0, len(points)):
            rv[i][0] = points[i][0]
            rv[i][1] = 1.0 - (points[i][0] + points[i][1]) # 1.0 - (x + y)
    elif cd is Direction.Increasing and cc is Concavity.Clockwise:
        for i in range(0, len(points)):
            rv[i][0] = points[i][0]
            rv[i][1] = points[i][1] - points[i][0] # y - x
    else:
        for i in range(0, len(points)):
            rv[i][0] = points[i][0]
            rv[i][1] = math.fabs(points[i][1] - points[i][0]) # abs(y - x)
    
    return rv


def single_knee(points, t, cd, cc):
    Ds = ema.linear(points, t)
    pmin = Ds.min(axis = 0)
    pmax = Ds.max(axis = 0)
    Dn = (Ds - pmin)/(pmax - pmin)
    Dd = differences(Dn, cd, cc)
    peaks = peak_detection.all_peaks(Dd)
    idx = peak_detection.highest_peak(points, peaks)
    if idx == -1:
        return None
    else:
        return idx


def knee(points, sensitivity = 1.0, cd=Direction.Decreasing, cc=Concavity.Clockwise, debug=False):
    Ds = ema.linear(points, 1.0)
    #print(Ds)

    pmin = Ds.min(axis = 0)
    pmax = Ds.max(axis = 0)
    Dn = (Ds - pmin)/(pmax - pmin)
    #print(Dn)

    Dd = differences(Dn, cd, cc)
    #print(Dd)

    idx = []
    lmxThresholds = []
    detectKneeForLastLmx = False
    knees=[]

    for i in range(1, len(Dd)-1):
        y0 = Dd[i-1][1]
        y = Dd[i][1]
        y1 = Dd[i+1][1]
        
        if y0 < y and y > y1:
            idx.append(i)
            tlmx = y - sensitivity / (len(Dd) - 1)
            lmxThresholds.append(tlmx)
            detectKneeForLastLmx = True
        
        if detectKneeForLastLmx:
            if y1 < lmxThresholds[-1]:
                knees.append(idx[-1])
                detectKneeForLastLmx = False

    if debug:
        return {'knees': np.array(knees),
        'Ds':Ds, 'Dn': Dn, 'Dd':Dd}        

    return np.array(knees)


def knee2(points, sensitivity, cd=Direction.Decreasing, cc=Concavity.Clockwise, debug=False):
    Ds = ema.linear(points, 1.0)
    #print(Ds)

    pmin = Ds.min(axis = 0)
    pmax = Ds.max(axis = 0)
    Dn = (Ds - pmin)/(pmax - pmin)
    #print(Dn)

    Dd = differences(Dn, cd, cc)
    #print(Dd)

    # Original Code
    idx = []
    lmxThresholds = []
    detectKneeForLastLmx = False
    knees=[]

    for i in range(1, len(Dd)-1):
        y0 = Dd[i-1][1]
        y = Dd[i][1]
        y1 = Dd[i+1][1]
        
        if y0 < y and y > y1:
            idx.append(i)
            tlmx = y - sensitivity / (len(Dd) - 1)
            lmxThresholds.append(tlmx)
            detectKneeForLastLmx = True
        
        if detectKneeForLastLmx:
            if y1 < lmxThresholds[-1]:
                knees.append(idx[-1])
                detectKneeForLastLmx = False

    # New version
    peaks_idx = peak_detection.all_peaks(Dd)

    significant_peaks_idx =  peak_detection.significant_peaks(Dd, peaks_idx, 0.25)
    
    knees_significant=[]
    for i in range(0, len(significant_peaks_idx)):
        if significant_peaks_idx[i]:
            knees_significant.append(i)

    significant_peaks_idx =  peak_detection.significant_zscore_peaks(Dd, peaks_idx)
    
    knees_z=[]
    for i in range(0, len(significant_peaks_idx)):
        if significant_peaks_idx[i]:
            knees_z.append(i)
    
    knees_iso = []
    significant_peaks_idx = peak_detection.significant_zscore_peaks_iso(Dd, peaks_idx)
    for i in range(0, len(significant_peaks_idx)):
        if significant_peaks_idx[i]:
            knees_iso.append(i)

    #print(knees)

    if debug:
        return {
        'knees_z': np.array(knees_z),
        'knees': np.array(knees),
        'knees_significant': np.array(knees_significant),
        'knees_iso': np.array(knees_iso),
        'Ds':Ds, 'Dn': Dn, 'Dd':Dd}        

    return np.array(knees)


def auto_knee(points, sensitivity=1.0, debug=False):
    start = points[0]
    end = points[-1]

    m = (end[1] - start[1]) / (end[0] - start[0])
    b = start[1] - (m * start[0])

    if m > 0.0:
        cd = Direction.Increasing
    else:
        cd = Direction.Decreasing
    
    #y = np.transpose(points)[1]
    #x = points[:,0]
    y = points[:,1]
    yhat = np.empty(len(points))
    for i in range(0, len(points)):
        yhat[i] = points[i][0]*m+b

    vote = np.sum(y - yhat)

    if cd is Direction.Increasing and vote > 0:
        cc = Concavity.Clockwise
    elif cd is Direction.Increasing and vote <= 0:
        cc = Concavity.Counterclockwise
    elif cd is Direction.Decreasing and vote > 0:
        cc = Concavity.Clockwise
    else:
        cc = Concavity.Counterclockwise
    
    knees_1 = knee2(points, sensitivity, cd, cc, debug)
    
    if cc is Concavity.Clockwise:
        knees_2 = knee2(points, sensitivity, cd, Concavity.Counterclockwise, debug)
    else:
        knees_2 = knee2(points, sensitivity, cd, Concavity.Clockwise, debug)


    if debug:
        # Merge results from dual kneedle
        keys = ['knees_z', 'knees', 'knees_significant', 'knees_iso']
        knees = knees_1

        for key in keys:
            tmp_1 = knees_1[key]
            tmp_2 = knees_2[key]
            tmp = np.concatenate((tmp_1, tmp_2))
            tmp = np.unique(tmp)
            tmp.sort
            knees[key] = tmp
    else:
        knees = np.concatenate((knees_1, knees_2))
        knees = np.unique(knees)
        knees.sort
         
    return knees


def get_knee(points, t=1):
    b, m = linear_fit_points(points)
    
    if m > 0.0:
        cd = Direction.Increasing
    else:
        cd = Direction.Decreasing

    y = points[:,1]
    yhat = np.empty(len(points))
    for i in range(0, len(points)):
        yhat[i] = points[i][0]*m+b

    vote = np.sum(y - yhat)

    if cd is Direction.Increasing and vote > 0:
        cc = Concavity.Clockwise
    elif cd is Direction.Increasing and vote <= 0:
        cc = Concavity.Counterclockwise
    elif cd is Direction.Decreasing and vote > 0:
        cc = Concavity.Clockwise
    else:
        cc = Concavity.Counterclockwise
    
    return single_knee(points, t, cd, cc)


def multi_knee(points, t1=0.99, t2=2):
    return mk.multi_knee(get_knee, points, t1, t2)
