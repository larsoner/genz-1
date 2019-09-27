#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" """

from mne.externals.h5io import read_hdf5

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


def highest_centrality(cent_dict):
    """Returns a tuple (node,value) with the node with largest value from
     Networkx centrality dictionary."""
    # Create ordered tuple of centrality data
    cent_items = [(b, a) for (a, b) in cent_dict.iteritems()]
    # Sort in descending order
    cent_items.sort()
    cent_items.reverse()
    return tuple(reversed(cent_items[0]))


def centrality_scatter(dict1, dict2, path="", ylab="", xlab="", title="",
                       line=False):
    """Draw scatter plot of two Networkx dictionaries """
    # Create figure and drawing axis
    figure, axis = plt.subplots(1, 1, figsize=(7, 7))
    # Create items and extract centralities
    items1 = sorted(dict1.items())
    items2 = sorted(dict2.items())
    xdata = [b for a, b in items1]
    ydata = [b for a, b in items2]
    # Add each actor to the plot by ID
    for p in range(len(items1)):
        axis.text(x=xdata[p], y=ydata[p], s=str(items1[p][0]), color="b")
    if line:
        # use NumPy to calculate the best fit
        mx, b = np.polyfit(xdata, ydata, 1)
        f = [b + mx * xi for xi in xdata]
        axis.plot(xdata, f, ls='--', color='g')
    # Set new x- and y-axis limits
    plt.xlim((0.0, max(xdata) + (.15 * max(xdata))))
    plt.ylim((0.0, max(ydata) + (.15 * max(ydata))))
    # Add labels and save
    ax1.set_title(title)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)
    plt.savefig(path)


h5 = read_hdf5(
    '/home/ktavabi/Github/genz/genz/data/genz105_9a_beta_fcDs.h5')
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

G = nx.from_numpy_matrix(adj.astype(int), parallel_edges=False)
pos = nx.spring_layout(G)

# Betweenness centrality - brokers/bridges
betweenness = nx.betweenness_centrality(G, normalized=False)
brokers = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
focus = [n for (n, _) in brokers[10::10]]
fig, ax = plt.subplots(1, 1, figsize=(8, 10))
fig.subplots_adjust(top=0.92, left=0.07, right=0.97,
                    hspace=0.3, wspace=0.3)
nx.draw(nx.subgraph(G, focus), pos=pos, with_labels=True, ax=ax)

# Eigenvector centrality - hubs
eigenvector = nx.eigenvector_centrality(G)
hubs = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)
focus = [n for (n, _) in hubs[10::10]]
fig, ax = plt.subplots(1, 1, figsize=(8, 10))
fig.subplots_adjust(top=0.92, left=0.07, right=0.97,
                    hspace=0.3, wspace=0.3)
nx.draw(nx.subgraph(G, focus), pos=pos, with_labels=True, ax=ax)

# Closeness
closeness = nx.closeness_centrality(G)
cl = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
focus = [n for (n, _) in cl[10::10]]
fig, ax = plt.subplots(1, 1, figsize=(8, 10))
fig.subplots_adjust(top=0.92, left=0.07, right=0.97,
                    hspace=0.3, wspace=0.3)
nx.draw(nx.subgraph(G, focus), pos=pos, with_labels=True, ax=ax)

# Local clustering
tris = nx.triangles(G)
tris = sorted(tris.items(), key=lambda x: x[1], reverse=True)
focus = [n for (n, _) in tris[10::10]]
fig, ax = plt.subplots(1, 1, figsize=(8, 10))
fig.subplots_adjust(top=0.92, left=0.07, right=0.97,
                    hspace=0.3, wspace=0.3)
nx.draw(nx.subgraph(G, focus), pos=pos, with_labels=True, ax=ax)

clustering = nx.clustering(G)
clustering = [(x, clustering[x]) for x in sorted(list(G.nodes),
                                                 key=lambda x: eigenvector[
                                                     x],
                                                 reverse=True)]
focus = [n for (n, _) in clustering[10::10]]
fig, ax = plt.subplots(1, 1, figsize=(8, 10))
fig.subplots_adjust(top=0.92, left=0.07, right=0.97,
                    hspace=0.3, wspace=0.3)
nx.draw(nx.subgraph(G, focus), pos=pos, with_labels=True, ax=ax)
