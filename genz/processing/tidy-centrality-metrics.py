#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path as op
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from h5io import read_hdf5

import mne


def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


ages = (9, 17)
freq_idx = [4, 5]  # np.arange(6)  # use all freq ranges
measures = ('betweeness', 'triangles')

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
        fast_fname = 'genz_%s_%s_fast.h5' % (age, measure)
        if not op.isfile(fast_fname):
            print('Converting %s measure %s' % (age, measure))
            data = read_hdf5(op.join('/Users/ktavabi/Github/Projects/genz/genz/data',
            'genz_%s_%s.h5' % (age, measure)))
            data = data['data_vars'][measure]['data']
            data = np.array(data)
            assert data.dtype == np.float
            write_hdf5(fast_fname, data)
        data = read_hdf5(fast_fname)
        if shape is None:
            shape = data.shape
            assert shape[-1] == 2
        assert data.shape == shape
        assert data.ndim == 4
        #data = data[freq_idx]  # only use these freqs
        # deal with reordering (undo it to restore original order)
        order = np.argsort(data[:, :, :, 0], axis=-1)
        data = data[..., 1]
        for ii in range(data.shape[0]):
            for jj in range(data.shape[1]):
                data[ii, jj] = data[ii, jj, order[ii, jj]]
        # put in subject, freq, roi order
        data = data.transpose(1, 0, 2)
        data = np.reshape(data, (len(data), -1))  # subjects, ...
        if mi == 0:
            y.extend([age] * len(data))
        X[ai].append(data)
    X[ai] = np.concatenate(X[ai], axis=-1)
y = np.array(y, float)
X = np.concatenate(X, axis=0)
print(X.shape, y.shape)
