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
    '/mnt/jaba/meg/genz_resting/genz105_9a/epochs/genz105_9a_beta_fcDs.h5')
# Select upper triangle of correlation matrix
corr = h5['corr']
# TODO threshold netowrk correlation matrices
cutoff = np.percentile(corr, 90)
ut, ui = np.unique(corr[corr > cutoff], return_index=True)
fig, ax = plt.subplots(figsize=(4, 4))
n, bins, patches = plt.hist(ut, 10, density=True, facecolor='g', alpha=0.75)
x1, x2 = np.percentile(ut, [90, 99.9])
plt.axvspan(x1, x2, facecolor='crimson', alpha=0.3)
plt.xlabel('pearson r\'s')
plt.ylabel('Probability')
plt.title('Histogram of correlations')
plt.xlim(ut.min(), ut.max())
plt.grid(True)
plt.show()

fig, ax = plt.subplots(figsize=(4, 4))
img = ax.imshow(ut, cmap='viridis',
                clim=np.percentile(ut, [0, 95]),
                interpolation='nearest', origin='lower')
fig.colorbar(img, ax=ax)
fig.tight_layout()

ma_ = h5['corr'] > np.percentile(h5['corr'], 90)
G = nx.from_numpy_array(h5['corr'][ma_])
fig, axes = plt.subplots(2, 1)
fig.subplots_adjust(top=0.92, left=0.07, right=0.97,
                    hspace=0.3, wspace=0.3)
ax1, ax2 = axes  # unpack the axes

nx.draw(G, with_labels=True, ax=ax1)

# Shortest path lengths between all nodes
paths = dict(nx.all_pairs_dijkstra_path_length(G))
sns.heatmap(pd.DataFrame(paths), cmap='terrain', ax=ax2)

# clustering
cluster = pd.DataFrame(dict(nx.algorithms.cluster.clustering(G)),
                       index=np.arange(len(G)))  # local
# nx.algorithms.cluster.triangles() for global

# efficiency
eff_l = nx.algorithms.local_efficiency(G)
eff_g = nx.algorithms.global_efficiency(G)

# small world
sigma = nx.algorithms.sigma(G)
omega = nx.algorithms.omega(G)

# betweenness centrality
bc = nx.algorithms.betweenness_centrality(G)

# resilience
re = nx.density(G)
