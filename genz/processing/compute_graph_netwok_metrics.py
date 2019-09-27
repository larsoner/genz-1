#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" """

from mne.externals.h5io import read_hdf5

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

h5 = read_hdf5(
    '/users/ktavabi/Data/genz105_9a_beta_fcDs.h5')
# adjacency matrix
corr = h5['corr']
mask = np.where(corr >= np.percentile(corr, 20), 1, 0)  # 20 percentile corr
adj = np.ma.masked_array(corr, mask).mask

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
n, bins, patches = ax1.hist(corr, 10, density=True, facecolor='c',
                            alpha=0.85)
ax1.set_xlabel('pearson r\'s')
ax1.set_ylabel('Probability')
ax1.grid(True)
ax2.set_title('correlation matrix')
img = ax2.imshow(corr, cmap='viridis',
                 clim=np.percentile(corr, [0, 95]),
                 interpolation='nearest', origin='lower')
fig.colorbar(img, ax=ax2)

G = nx.from_numpy_matrix(adj, parallel_edges=False)
pos = nx.spring_layout(G)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.subplots_adjust(top=0.92, left=0.07, right=0.97,
                    hspace=0.3, wspace=0.3)
nx.draw(G, with_labels=True, pos=pos, ax=ax1)

# Shortest path lengths between all nodes
paths = dict(nx.all_pairs_dijkstra_path_length(G))
sns.heatmap(pd.DataFrame(paths), cmap='viridis', ax=ax2)

# clustering
# relationships between a node's neighbors, rather than those of the node
# itself, i.e., whether a node's neighbors are connected to each other (
# transitivity). The result of such relationships are triangles: three nodes,
# all mutually connected. The tendency for such triangles to arise is called
# clustering. When strong clustering is present, it often suggests robustness,
# and redundancy in a network.
clustering = nx.clustering(G)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
n, bins, patches = ax.hist(clustering.values(), 10, density=True,
                           facecolor='c',
                           alpha=0.85)
ax.set_xlabel('cluster size')
ax.set_ylabel('Probability')
ax.grid(True)
ax.set_title('clustering')

# efficiency
eff_l = [nx.efficiency(G, u, v) for u, v in G.edges]

# small world
sigma = nx.sigma(G)
omega = nx.algorithms.omega(G)

# betweenness centrality
bc = nx.algorithms.betweenness_centrality(G)

# resilience
re = nx.density(G)
