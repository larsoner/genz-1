#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Write age cohort Graph Theoretical centrality metrics to disk"""

import os.path as op

import mne
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from mne.externals.h5io import read_hdf5, write_hdf5

from genz import defaults


def sort_dict(dd, reverse=True):
    """"""
    return sorted(dd.items(), key=lambda x: x[1], reverse=reverse)


# Load labels
fslabels = mne.read_labels_from_annot('fsaverage',
                                      'aparc_sub', 'both',
                                      subjects_dir=defaults.subjects_dir)
fslabels = [label for label in fslabels
            if not label.name.startswith('unknown')]
label_nms = [rr.name for rr in fslabels]
picks = pd.read_csv(op.join(defaults.static, 'picks.tsv'), sep='\t')
picks.drop(picks[picks.id.isin(defaults.exclude)].index, inplace=True)
picks.sort_values(by='id', inplace=True)
dt = np.dtype('int, float')
dims = ['band', 'subject', 'roi']

for aix, age in enumerate(defaults.ages):
    subjects = ['genz%s' % ss for ss in picks[picks.ag == age].id]
    coords = [['beta'], [subjects], np.arange(len(label_nms))]
    for ix, band in enumerate(defaults.bands.keys()):
        for iix, subj in enumerate(subjects):
            h5 = read_hdf5(
                op.join(defaults.datadir, '%s_%s_fcDs.h5' % (subj, band)))
            # adjacency matrix
            corr = h5['corr']
            mask = np.where(corr >= np.percentile(corr, 20), 1, 0)
            adj = np.ma.masked_array(corr, mask).mask
            G = nx.from_numpy_matrix(adj.astype(int), parallel_edges=False)
            if ix & iix == 0:
                betweeness = np.zeros((1, 1,
                                       len(label_nms)),
                                      dtype=dt)
                betweeness.dtype.names = ['label', 'value']
                hubness = np.zeros_like(betweeness)
                closeness = np.zeros_like(betweeness)
                triangles = np.zeros_like(betweeness)
                clustering = np.zeros_like(betweeness)
            # Betweenness centrality - brokers/bridges
            betweeness[ix, iix] = np.array(sort_dict(
                nx.betweenness_centrality(G, normalized=False)), dtype=dt)
            # Eigenvector centrality - hubs
            eigenvector = sort_dict(nx.eigenvector_centrality(G))
            hubness[ix, iix] = np.array(eigenvector, dtype=dt)
            # Closeness
            closeness[ix, iix] = np.array(sort_dict(nx.closeness_centrality(G)),
                                          dtype=dt)
            # Local clustering
            triangles[ix, iix] = np.array(sort_dict(nx.triangles(G)), dtype=dt)
            clusters = nx.clustering(G)
            clustering[ix, iix] = np.array([(x, clusters[x]) for x in
                                            sorted(list(G.nodes),
                                                   key=lambda x: eigenvector[x],
                                                   reverse=True)], dtype=dt)
    # bands x subjects x labels, x valaue nd-array
    betweeness_ = xr.DataArray(betweeness, coords=coords, dims=dims)
    write_hdf5(op.join(defaults.datadir, 'genz_%s_betweeness.h5' % age),
               betweeness_.to_dict())
