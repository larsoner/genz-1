#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as op
import numpy as np
from autoreject import AutoReject
import mne
from mne import (
    read_epochs, compute_rank
    )
from mne.cov import regularize
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from meeg_preprocessing import config
from genz import defaults


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


def extract_labels_timeseries(subj, r, highpass, lowpass, cov, fwd,
                              subjects_dir, fname, labels, return_generator):
    # epoch raw into 5 sec trials
    events = mne.make_fixed_length_events(r, duration=5.)
    epochs = mne.Epochs(r, events=events, tmin=0, tmax=5.,
                        baseline=None, reject=None, preload=True)
    if not op.isfile(fname):
        # k-fold CV thresholded artifact rejection
        ar = AutoReject()
        epochs = ar.fit_transform(epochs)
        print('      \nSaving ...%s' % op.relpath(fname,
                                                  defaults.megdata))
        epochs.save(fname, overwrite=True)
    epochs = read_epochs(fname)
    print('%d, %d (Epochs, drops)' %
          (len(events), len(events) - len(epochs.selection)))
    # epochs.plot_psd()
    idx = np.setdiff1d(np.arange(len(events)), epochs.selection)
    # r = r.copy().filter(lf, hf, fir_window='blackman',
    #                       method='iir', n_jobs=config.N_JOBS)
    iir_params = dict(order=4, ftype='butter', output='sos')
    epochs_ = epochs.copy().filter(highpass, lowpass, method='iir',
                                   iir_params=iir_params,
                                   n_jobs=config.N_JOBS)
    # epochs_.plot_psd(average=True, spatial_colors=False)
    mne.Info.normalize_proj(epochs_.info)
    # epochs_.plot_projs_topomap()
    # regularize covariance
    # rank = compute_rank(cov, rank='full', info=epochs_.info)
    cov = regularize(cov, r.info)
    inv = make_inverse_operator(epochs_.info, fwd, cov)
    # Compute label time series and do envelope correlation
    stcs = apply_inverse_epochs(epochs_, inv, lambda2=1. / 9.,
                                pick_ori='normal',
                                return_generator=True)
    morphed = mne.morph_labels(labels, subj,
                               subjects_dir=subjects_dir)
    return mne.extract_label_time_course(stcs, morphed, fwd['src'],
                                         return_generator=return_generator,
                                         verbose=True)
