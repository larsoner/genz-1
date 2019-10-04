#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Write age cohort Graph Theoretical global metrics to disk"""

import math
import os.path as op
import time

import networkx as nx
import xarray as xr
import numpy as np
import pandas as pd
from mne.externals.h5io import read_hdf5
from mnefun._mnefun import timestring

from genz import defaults


def entropy(x):
    """return the entropy of a list of numbers.
    same as scipy.stats.entropy
    """
    # Normalize
    total = sum(x)
    x = [xi / total for xi in x]
    H = sum([-xi * math.log2(xi) for xi in x])
    return H


def get_globals(G):
    return (nx.average_shortest_path_length(G),
            nx.diameter(G),
            nx.transitivity(G),
            nx.average_clustering(G),  # Global clustering coeff
            nx.density(G),  # resilience
            nx.node_connectivity(G),  # minimum min-cut to disconnect
            nx.average_node_connectivity(G),
            entropy(nx.eigenvector_centrality(G).values()))


picks = pd.read_csv(op.join(defaults.static, 'picks.tsv'), sep='\t')
picks.drop(picks[picks.id.isin(defaults.exclude)].index, inplace=True)
picks.sort_values(by='id', inplace=True)
dt = np.dtype('int, float')
bands = list(defaults.bands.keys())
metrics = ['char_len', 'diameter', 'transitivity',
           'clustering', 'density',
           'min_cut', 'connectivity', 'entropy']
dims = ['band', 'subject', 'metric']

for aix, age in enumerate(defaults.ages):
    subjects = ['genz%s' % ss for ss in picks[picks.ag == age].id]
    print('  Loading %s years-old data.' % age)
    t0 = time.time()
    for ix, band in enumerate(bands):
        print('     Computing %s band global graph measures...' % band)
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
                metric_ds = np.zeros((len(bands), len(subjects),
                                      len(metrics)))
            metric_ds[ix, iix] = get_globals(G)
    # bands x subjects x metric
    foo = xr.DataArray(metric_ds, coords=[['gamma'], [1], metrics], dims=dims)
    foo.to_netcdf(op.join(defaults.datadir, 'genz_%s_globals.nc' % age))
    print('Run time %s ' % timestring(time.time() - t0))
    # bar = foo.to_dataframe().unstack('metric')
    # bar.columns.set_levels(metrics, level=1, inplace=True)
    # bar.droplevel(level=0, axis=1).to_csv(
    #    op.join(defaults.datadir, 'genz_%s_globals.csv' % age))
