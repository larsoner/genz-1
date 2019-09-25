#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" """

from mne.externals.h5io import read_hdf5

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

arr = read_hdf5('/Users/ktavabi/Data/genz105_9a_beta_fcDs.h5')['corr']
G = nx.subgraph(nx.from_numpy_array(arr), np.arange(20))
G = nx.from_numpy_array(arr)


fig, axes = plt.subplots(2, 2)
fig.subplots_adjust(top=0.92, left=0.07, right=0.97,
                    hspace=0.3, wspace=0.3)
((ax1, ax2), (ax3, ax4)) = axes  # unpack the axes

nx.draw(G, with_labels=True, ax=ax1)
nx.draw(G.to_directed(), with_labels=True, ax=ax2)

# Shortest path lengths between all nodes
paths = [dict(nx.all_pairs_dijkstra_path_length(g)) for g in [G,
                                                              G.to_directed()]]
for ax, pth in zip([ax3, ax4], paths):
    sns.heatmap(pd.DataFrame(pth), cmap='terrain', ax=ax)

# clustering
cluster = pd.DataFrame(dict(nx.algorithms.cluster.clustering(G)),
                       index=np.arange(len(G)))  # local
# nx.algorithms.cluster.triangles() for global

# efficiency
eff_l = nx.algorithms.local_efficiency(G)
eff_g= nx.algorithms.global_efficiency(G)

# betweenness centrality
bc = nx.algorithms.betweenness_centrality(G)

tutte = nx.tutte_graph()
