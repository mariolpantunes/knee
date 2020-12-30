# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import numpy as np
import math


def distance_to_similarity(array: np.ndarray) -> np.ndarray:
    return max(array) - array


def rank(array: np.ndarray) -> np.ndarray:
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks


def curvature_ranking(points: np.ndarray, knees: np.ndarray, relative=True) -> np.ndarray:
    pt = np.transpose(points)

    x = pt[0]
    y = pt[1]

    gradient1 = np.gradient(y, x, edge_order=1)
    gradient2 = np.gradient(y, x, edge_order=2)

    rankings = []
    for idx in knees:
        curvature = math.fabs(gradient2[idx]) / (1.0 + gradient1[idx]**2.0)**(1.5)
        rankings.append(curvature)
    
    if relative:
        rankings = rank(np.array(rankings))
    rankings = (rankings - np.min(rankings))/np.ptp(rankings)

    return rankings


def menger_curvature(f, g, h):
    x1 = f[0]
    y1 = f[1]
    x2 = g[0]
    y2 = g[1]
    x3 = h[0]
    y3 = h[1]

    nom = 2.0 * math.fabs((x2-x1)*(y3-y2))-((y2-y1)*(x3-x2))
    temp = math.fabs((x2-x1)**2.0 + (y2-y1)**2.0)*math.fabs((x3-x2)**2.0 + (y3-y2)**2.0) * math.fabs((x1-x3)**2.0 + (y1-y3)**2.0)
    dem = math.sqrt(temp)

    return nom/dem


def menger_curvature_ranking(points: np.ndarray, knees: np.ndarray, relative=True) -> np.ndarray:
    rankings = []
    for idx in knees:
        f = points[idx]
        g = points[idx-1]
        h = points[idx+1]
        curvature = menger_curvature(f, g, h)
        rankings.append(curvature)
    
    if relative:
        rankings = rank(np.array(rankings))
    rankings = (rankings - np.min(rankings))/np.ptp(rankings)

    return rankings

def l_ranking(points: np.ndarray, knees: np.ndarray, relative=True) -> np.ndarray:
    rankings = []

    pt = np.transpose(points)

    x = pt[0]
    y = pt[1]

    coef_left, r_left, *other  = np.polyfit(x[0:knees[0]+1], 
    y[0:knees[0]+1], 1, full=True)
    coef_right, r_rigth, *other = np.polyfit(x[knees[0]:knees[1]+1],
    y[knees[0]:knees[1]+1], 1, full=True)
    error = (r_left[0] + r_rigth[0]) / 2.0
    rankings.append(error)

    for i in range(1, len(knees)-1):
        coef_left, r_left, *other  = np.polyfit(x[knees[i-1]:knees[i]+1], 
        y[knees[i-1]:knees[i]+1], 1, full=True)
        coef_right, r_rigth, *other = np.polyfit(x[knees[i]:knees[i+1]+1],
        y[knees[i]:knees[i+1]+1], 1, full=True)
        error = (r_left[0] + r_rigth[0]) / 2.0
        rankings.append(error)
    
    coef_left, r_left, *other  = np.polyfit(x[knees[-2]:knees[-1]+1], 
    y[knees[-2]:knees[-1]+1], 1, full=True)
    coef_right, r_rigth, *other = np.polyfit(x[knees[-1]:],
    y[knees[-1]:], 1, full=True)
    error = (r_left[0] + r_rigth[0]) / 2.0
    rankings.append(error)

    rankings = distance_to_similarity(np.array(rankings))
    
    if relative:
        rankings = rank(np.array(rankings))
    rankings = (rankings - np.min(rankings))/np.ptp(rankings)

    return rankings


def slopes_to_angle(m1: float, m2: float) -> float:
    tan = (m1-m2)/(1.0+m1*m2)
    angle_positive = math.atan(tan)
    angle_negative = math.atan(-tan)

    print('Positive: {}'.format(angle_positive))
    print('Negative: {}'.format(angle_negative))

    if angle_positive < angle_negative:
        return angle_positive
    else:
        return angle_negative


def isodata(array: np.ndarray) -> float:
    mean = array.mean()
    previous_mean = 0 
    
    while mean != previous_mean:
        mean_left = array[array <= mean].mean()
        mean_right = array[array > mean].mean()

        previous_mean = mean
        mean = (mean_left+mean_right) / 2.0

    return mean

def dfdt_ranking(points: np.ndarray, knees: np.ndarray, relative=True) -> np.ndarray:
    pt = np.transpose(points)

    x = pt[0]
    y = pt[1]

    gradient1 = np.gradient(y, x, edge_order=1)
    t = isodata(gradient1)

    rankings = []
    for idx in knees:
        error = math.fabs(t - gradient1[idx])
        rankings.append(error)
    
    rankings = distance_to_similarity(np.array(rankings))

    if relative:
        rankings = rank(np.array(rankings))
    rankings = (rankings - np.min(rankings))/np.ptp(rankings)

    return rankings


def angle_ranking(points: np.ndarray, knees: np.ndarray, neighborhood=30, relative=True) -> np.ndarray:
    rankings = []

    pt = np.transpose(points)

    x = pt[0]
    y = pt[1]

    for idx in knees:
        coef_left, r_left, *other  = np.polyfit(x[idx-neighborhood:idx+1], 
        y[idx-neighborhood:idx+1], 1, full=True)
        coef_right, r_rigth, *other = np.polyfit(x[idx:idx+neighborhood+1],
        y[idx:idx+neighborhood+1], 1, full=True)
        print('-----')
        print(coef_left)
        print(coef_right)
        angle = slopes_to_angle(coef_left[0], coef_right[0])
        print(angle)
        print('-----')
        error = (r_left[0] + r_rigth[0]) / 2.0
        rankings.append(error)

    rankings = distance_to_similarity(np.array(rankings))
    
    if relative:
        rankings = rank(np.array(rankings))
    rankings = (rankings - np.min(rankings))/np.ptp(rankings)

    return rankings

def slope_ranking(points: np.ndarray, knees: np.ndarray, t=0.9, relative=True) -> np.ndarray:
    rankings = []

    pt = np.transpose(points)

    x = pt[0]
    y = pt[1]
    
    j = 0
    r = (np.corrcoef(x[j:knees[0]+1], y[j:knees[0]+1])[0,1])**2.0
    while r < t:
        j = int((j+knees[0])/2.0)
        r = (np.corrcoef(x[j:knees[0]+1], y[j:knees[0]+1])[0,1])**2.0
    print('R2({}) = {}'.format(0, r))
    
    slope = (y[j]-y[knees[0]]) / (x[j]-x[knees[0]])
    rankings.append(math.fabs(slope))
    
    for i in range(1, len(knees)):
        j = knees[i-1]
        r = (np.corrcoef(x[j:knees[i]+1], y[j:knees[i]+1])[0,1])**2.0
        while r < t:
            j = int((j + knees[i])/2.0)
            r = (np.corrcoef(x[j:knees[i]+1], y[j:knees[i]+1])[0,1])**2.0
        print('R2({}) = {}'.format(i, r))
        slope = (y[j]-y[knees[0]]) / (x[j]-x[knees[0]])
        rankings.append(math.fabs(slope))

    if relative:
        rankings = rank(np.array(rankings))
    rankings = (rankings - np.min(rankings))/np.ptp(rankings)

    return rankings