# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import numpy as np


'''def strait_line(points, a, b, t=0.8):
    # setup
    pt = np.transpose(points)
    x = pt[0]
    y = pt[1]
    i = previous_i = a

    # compute initial value
    r2 = (np.corrcoef(x[i:b+1], y[i:b+1])[0,1])**2.0

    scores = []

    for i in range(a, b):
        r2 = (np.corrcoef(x[i:b+1], y[i:b+1])[0,1])**2.0
        scores.append(r2)
    scores = np.array(scores)
    idx = (np.abs(scores - t)).argmin()

    return a+idx'''

def strait_line(points, a, b, t=0.8):
    #corner cases
    if abs(a - b) <= 1:
        return a
    # setup
    x = points[:,0]
    y = points[:,1]
    i = a
    right = b

    # compute initial value
    r2 = (np.corrcoef(x[i:b+1], y[i:b+1])[0,1])**2.0

    if r2 >= t:
        return a
    else:
        print('[{}, {}] {}'.format(i, right, r2))
        i = int((i+right)/2.0)

        while abs(i-right) > 1:
            r2 = (np.corrcoef(x[i:b+1], y[i:b+1])[0,1])**2.0
            
            if r2 < t:
                i = int((i+right)/2.0)
            else:
                right = i
                i = int((a+i)/2.0)
            print('[{}, {}] {}'.format(i, right, r2))
        
        if right != b:
            return right
        else:
            return i


def perpendicular_distance2(pt, start, end):
    return np.fabs(np.cross(end-start,pt-start)/np.linalg.norm(end-start))

def rdp(points, r=0.95):
    end = len(points) - 1
    d = perpendicular_distance2(points, points[0], points[end])
    index = np.argmax(d)

    m = (points[end][1]-points[0][1])/(points[end][0]-points[0][0])
    b = points[0][1]- (m * points[0][0])
    y = np.transpose(points)[1]
    yhat = np.empty(len(y))
    for i in range(0, end+1):
        yhat[i] = points[i][0]*m+b
    
    ybar = np.sum(y)/len(y)          
    ssreg = np.sum((y-yhat)**2)
    sstot = np.sum((y - ybar)**2)

    if sstot > 0.0:
        determination = 1.0 - (ssreg / sstot)
    else:
        determination = 1.0 - ssreg

    if determination < r :
        left, left_points = rdp(points[0:index+1], r)
        right, right_points = rdp(points[index:end+1], r)
        points_removed = np.concatenate((left_points, right_points), axis=0)
        return np.concatenate((left[0:len(left)-1], right)), points_removed
    else:
        rv = np.empty([2,2])
        rv[0] = points[0]
        rv[1] = points[end]
        middle_point = (points[0][0] + points[end][0])/2.0
        points_removed = np.array([[middle_point, float(len(points) - 2.0)]])
        return rv, points_removed


#points = np.array([[0.0, 0.0], [1.0, 2.0], [1.2, 4.0], [2.3, 6], [2.9, 8], [5, 10]])
#sp = rdp(points)
#print(sp)