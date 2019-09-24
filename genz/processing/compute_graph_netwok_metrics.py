#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
- Undirected graph no direction in the edges.
- Edges in an undirected graph are not ordered pairs.
- Undirected graphs can be used to represent symmetric relationships b/n nodes.
- In a directed graph an edge is an ordered pair.
- In- and out-degree of each node in an undirected graph is equal.
- Matrix rep of undirected graph yields a symmetric graph
- Undirected can be converted to directed not vise versa.
- In an adjacency matrix, directed 2->3 means adj[i][j]=true
but adj[i][j]=false. In undirected it means adj[2][3]=adj[3][2]=true.
- A bipartite graph (or bigraph) is a graph whose vertices can be divided into
two disjoint and independent sets U and V such that every edge connects a
vertex in U to one in V. Vertex sets U and V are usually called the parts of
the graph. Equivalently, a bipartite graph is a graph that does not contain any
odd-length cycles.
"""

from mne.externals.h5io import read_hdf5

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


arr = read_hdf5('/mnt/jaba/meg/genz_resting/genz105_9a/epochs'
                '/genz105_9a_beta_fcDs.h5')['corr']
dg = nx.from_numpy_array(arr)
fig, axes = plt.subplots(2, 2)
fig.subplots_adjust(top=0.92, left=0.07, right=0.97,
                    hspace=0.3, wspace=0.3)
((ax1, ax2), (ax3, ax4)) = axes  # unpack the axes

nx.draw(nx.subgraph(dg, np.arange(10)), with_labels=True, ax=ax1)
ug = dg.to_directed()
nx.draw(nx.subgraph(ug, np.arange(10)), with_labels=True, ax=ax2)

# Shortest path lengths between all nodes
path = dict(nx.all_pairs_dijkstra_path_length(ug))
df = pd.DataFrame(path)
sns.heatmap(df, cmap='terrain', ax=ax4)

tutte = nx.tutte_graph()