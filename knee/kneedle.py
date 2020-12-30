# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import math
import numpy as np
from uts import ema
from enum import Enum
from zscore import zscore_points
#from rdp import rdp


class Direction(Enum):
    Increasing=0
    Decreasing=1


class Concavity(Enum):
    Counterclockwise=0
    Clockwise=1


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
            tlmx = y - sensitivity / (len(Dd) - 1);
            lmxThresholds.append(tlmx)
            detectKneeForLastLmx = True
        
        if detectKneeForLastLmx:
            if y1 < lmxThresholds[-1]:
                knees.append(idx[-1])
                detectKneeForLastLmx = False

    if debug:
        return {'knees': knees,
        'Ds':Ds, 'Dn': Dn, 'Dd':Dd}        

    return np.array(knees)


def find_next_tau(points, i, tau):
    #print('Find Next Tau')
    #print('Tau = {}'.format(tau))
    durations = points[i:,0] - points[i-1:-1,0]
    #print(durations)
    cumulative_durations = np.cumsum(durations)
    #print(cumulative_durations)
    #idx = cumulative_durations[cumulative_durations>tau]
    idx = np.argmax(cumulative_durations>tau)
    #print('IDX = {}'.format(idx))
    if idx == 0:
        return len(points)-1
    rv = i+idx
    #print('Next idex = {}'.format(rv))
    #input("Press Enter to continue...")
    return rv


def knee2(points, threshold, cd=Direction.Decreasing, cc=Concavity.Clockwise, debug=False):
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

    scores = []
    scores_left = []
    scores_right = []

    for i in range(1, len(Dd)-2):
        y0 = Dd[i-1][1]
        y = Dd[i][1]
        y1 = Dd[i+1][1]

        # check zscore
        if len(knees) == 0:
            tau = Dd[i][0] - Dd[0][0]
            left = math.fabs(zscore_points(y, Dd[0:i+1]))
            
        else:
            tau = Dd[i][0] - Dd[knees[-1]][0]
            left = math.fabs(zscore_points(y, Dd[knees[-1]:i+1]))
            
        j = find_next_tau(Dd, i, tau)

        if len(knees) == 0:
            score = math.fabs(zscore_points(y, Dd[0:j+1]))
        else:
            score = math.fabs(zscore_points(y, Dd[knees[-1]:j+1]))
        
        right = math.fabs(zscore_points(y, Dd[i:j+1]))
        
        scores_left.append([Dd[i][0], left])
        scores_right.append([Dd[i][0], right])
        scores.append([Dd[i][0], score])
        
        if y0 < y and y > y1 or y0 > y and y < y1:
            print('Tau = {} Previous index {} Next index {}'.format(tau, i,j))
            print('Z-score = {} and {}'.format(left, right))
            print('Z-score = {} ({})'.format(score, threshold))
            if left > threshold and right > threshold:
                knees.append(i)
            
            
            
            #
            #if score > threshold:
            #    knees.append(i)
            #idx.append(i)
            #tlmx = y - sensitivity / (len(Dd) - 1);
            #lmxThresholds.append(tlmx)
            #detectKneeForLastLmx = True
        
        
        #if detectKneeForLastLmx:
        #    if y1 < lmxThresholds[-1]:
        #        knees.append(idx[-1])
        #        detectKneeForLastLmx = False

    if debug:
        return {'knees': knees,
        'Ds':Ds, 'Dn': Dn, 'zscores': np.array(scores),
        'zscores_left': np.array(scores_left),
        'zscores_right': np.array(scores_right), 'Dd':Dd}        

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
    
    y = np.transpose(points)[1]
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
    
    return knee2(points, sensitivity, cd, cc, debug)

#points = np.array([[0.0, 0.0], [1.0, 2.0], [1.2, 4.0], [2.3, 6], [2.9, 8], [5, 10]])

#l = [[1.0, 1.0],[2, 0.25],[3, 0.111],[4, 0.0625],[5, 0.04],[6, 0.0277777],[7, 0.0204],[8, 0.015625],[9, 0.012345679],[10, .01]]
#points = np.array(l)
#print(points)

#knees = knee(points, 1.0, Direction.Decreasing, Concavity.Counterclockwise)
#print("Knee:", knees)

#knees = auto_knee(points)
#print("Auto Knee:", knees)