#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path as op
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from h5io import read_hdf5

import mne

from genz import defaults

ages = (9, 17)
freq_idx = [4, 5]  # np.arange(6)  # use all freq ranges
measures = ('betweeness', 'triangles')
data_dir = op.join(defaults.datadir)

freqs = ['beta', 'gamma']
labels = mne.read_labels_from_annot('fsaverage', 'aparc_sub')
labels = [label for label in labels if 'unknown' not in label.name]
rois = [ll.name for ll in labels]

###############################################################################
# Load data
# ---------

X, y = [list() for _ in range(len(ages))], list()
for ai, age in enumerate(ages):
    shape = None
    for mi, measure in enumerate(measures):
        fname = op.join(data_dir, 'genz_%s_%s.h5' % (age, measure))
        data = read_hdf5(fname)
        data = data['data_vars'][measure]['data']
        data = np.array(data)
        assert data.dtype == np.float
        if shape is None:
            shape = data.shape
            assert shape[-1] == 2
        assert data.shape == shape
        assert data.ndim == 4
        if mi == 0:
            y.extend([age] * len(data))
        X[ai].append(data)
    X[ai] = np.concatenate(X[ai], axis=-1)
y = np.array(y, float)
X = np.concatenate(X, axis=0)
print(X.shape, y.shape)

