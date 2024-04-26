#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '1.0'
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

import kneeliverse.kneedle as kneedle


#plt.style.use('seaborn-v0_8-paper')
plt.style.use('tableau-colorblind10')
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['lines.linewidth'] = 2


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

    # Color Blind adjusted colors and markers
    colormap=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', 
    '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']
    markers=['o', '*', '.', 'x', '+', 's', 'd', 'h', 'v']
    lines=['-', ':', '--', '-.']

    for i in range(k):
        cluster_i_idx = np.where(kmeans.labels_ == i)[0]
        X_cluster_i = X[cluster_i_idx]
        plt.scatter(X_cluster_i[:,2], X_cluster_i[:,3], c=colormap[i], s=40, marker=markers[i])
    
    ax = plt.gca()
    ax.set_xlabel('Petal length')
    ax.set_ylabel('Petal width')
    ax.set_xlim([0, 7.1])
    ax.set_ylim([0, 2.6])
    plt.savefig('out/kmeans_clusters.png', bbox_inches='tight', transparent=True)
    plt.savefig('out/kmeans_clusters.pdf', bbox_inches='tight', transparent=True)
    plt.show()

    plt.plot(wcss[:,0], wcss[:,1], color = colormap[0], linestyle=lines[0])
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS') #within cluster sum of squares
    plt.axvline(x = k, color = colormap[1], linestyle=lines[1])
    ax = plt.gca()
    ax.set_xlim([0, 13])
    ax.set_ylim([0, 160])
    plt.savefig('out/kmeans_wcss.png', bbox_inches='tight', transparent=True)
    plt.savefig('out/kmeans_wcss.pdf', bbox_inches='tight', transparent=True)
    plt.show()


if __name__ == '__main__':
    main()