#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as op
import itertools
import mne
import numpy as np


from genz import defaults


def expand_grid(data_dict):
    import pandas as pd
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


def compute_adjacency_matrix(connectivity, threshold_prop=0.2):
    """Threshold and binarize connectivity matrix.
    Notes:
        https://github.com/mne-tools/mne-python/blob
        /9585e93f8c2af699000bb47c2d6da1e9a6d79251/mne/connectivity/utils.py
        #L69-L92
    """
    if connectivity.ndim != 2 or \
            connectivity.shape[0] != connectivity.shape[1]:
        raise ValueError('connectivity must be have shape (n_nodes, n_nodes), '
                         'got %s' % (connectivity.shape,))
    n_nodes = len(connectivity)
    if np.allclose(connectivity, connectivity.T):
        split = 2.
        connectivity[np.tril_indices(n_nodes)] = 0
    else:
        split = 1.
    threshold_prop = float(threshold_prop)
    if not 0 < threshold_prop <= 1:
        raise ValueError('threshold must be 0 <= threshold < 1, got %s'
                         % (threshold_prop,))
    degree = connectivity.ravel()  # no need to copy because np.array does
    degree[::n_nodes + 1] = 0.
    n_keep = int(round((degree.size - len(connectivity)) *
                       threshold_prop / split))
    degree[np.argsort(degree)[:-n_keep]] = 0
    degree.shape = connectivity.shape
    if split == 2:
        degree += degree.T  # normally unsafe, but we know where our zeros are
    return degree

