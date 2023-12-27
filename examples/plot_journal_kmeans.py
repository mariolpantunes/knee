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
import numpy as np
import matplotlib.pyplot as plt 

from sklearn import datasets
from sklearn.cluster import KMeans

import knee.kneedle as kneedle


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.autolayout'] = True


logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    wcss = []

    for i in tqdm.tqdm(range(2, 13)):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init='auto', random_state=42)
        kmeans.fit(X)
        wcss.append((i, kmeans.inertia_))

    wcss = np.array(wcss)
    knee_idx = kneedle.knee(wcss)
    k = int(wcss[knee_idx][0])
    #k=3

    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    kmeans.fit(X)

    colormap=np.array(['#4C72B0','#DD8452','#55A868','#C44E52',
    '#8172B3','#937860','#DA8BC3','#8C8C8C','#CCB974','#64B5CD'])
    plt.scatter(X[:,2], X[:,3], c=colormap[kmeans.labels_], s=40)
    ax = plt.gca()
    ax.set_xlabel('Petal length')
    ax.set_ylabel('Petal width')
    plt.savefig('out/kmeans_clusters.png', bbox_inches='tight')
    plt.savefig('out/kmeans_clusters.pdf', bbox_inches='tight')
    plt.show()

    plt.plot(wcss[:,0], wcss[:,1], color = colormap[0])
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS') #within cluster sum of squares
    plt.axvline(x = k, color = colormap[1])
    plt.savefig('out/kmeans_wcss.png', bbox_inches='tight')
    plt.savefig('out/kmeans_wcss.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()