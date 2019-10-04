#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Write age cohort Graph Theoretical centrality metrics to disk"""

import os.path as op

import mne
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
import time

from mne.externals.h5io import read_hdf5, write_hdf5
from mnefun._mnefun import timestring
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
bands = list(defaults.bands.keys())

for aix, age in enumerate(defaults.ages):
    subjects = ['genz%s' % ss for ss in picks[picks.ag == age].id]
    print('  Loading %s years-old data.' % age)
    coords = [bands, subjects, np.arange(len(label_nms))]
    t0 = time.time()
    for ix, band in enumerate(bands):
        print('     Computing %s band centrality measures...' % band)
        for iix, subject in enumerate(subjects):
            print('     %s' % subject)
            subj_dir = op.join(defaults.megdata, subject)
            eps_dir = op.join(subj_dir, 'epochs')
            h5 = read_hdf5(
                op.join(eps_dir, '%s_%s_fcDs.h5' % (subject, band)))
            # adjacency matrix
            corr = h5['corr']
            mask = np.where(corr >= np.percentile(corr, 20), 1, 0)
            adj = np.ma.masked_array(corr, mask).mask
            G = nx.from_numpy_matrix(adj.astype(int), parallel_edges=False)
            if ix == 0 and iix == 0:
                betweeness = np.zeros((len(bands), len(subjects),
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
    for ds, nm in zip([betweeness, hubness, closeness, triangles, clustering],
                      ['betweeness', 'hubness', 'closeness', 'triangles',
                       'clustering']):
        foo = xr.Dataset({nm: (dims, ds)}).to_dict()
        for kk in ['coords', 'dims', 'attrs']:
            foo.pop(kk)
            if kk == 'attrs':
                foo['data_vars'][nm].pop('attrs')
        write_hdf5(op.join(defaults.datadir, 'genz_%s_%s.h5' % (age, nm)),
                   foo, overwrite=True)
    print('Run time %s ' % timestring(time.time() - t0))
