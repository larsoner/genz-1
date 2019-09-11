#!/usr/bin/env python

"""compute_coh_connectivity.py: compute functional connectivity (degree) using
narrow band envolope power.
    Does per age per subject:
        1. epoch raw into 5 sec trials
        2. compute erm covariance
        3. use Autoreject to clean up wide band epoched data
        per frequency band:
        4. regularize covariance
        5. invert and compute pairwise ROI envelope correlations
        6. compute functional (degree) connectivity
"""

__author__ = 'Kambiz Tavabi'
__copyright__ = 'Copyright 2018, Seattle, Washington'
__credits__ = ['Eric Larson']
__license__ = 'MIT'
__version__ = '1.0.1'
__maintainer__ = 'Kambiz Tavabi'
__email__ = 'ktavabi@uw.edu'

import os
import os.path as op

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import xarray as xr
from autoreject import AutoReject
from meeg_preprocessing import config
from meeg_preprocessing import utils
from mne import (
    read_epochs, compute_raw_covariance,
    compute_rank
    )
from mne.connectivity import envelope_correlation
from mne.cov import regularize
from mne.filter import next_fast_len
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mnefun import get_fsaverage_medial_vertices

from genz import defaults

utils.setup_mpl_rcparams(font_size=10)

new_sfreq = 200.
n_fft = next_fast_len(int(round(4 * new_sfreq)))
lims = [75, 85, 95]
medial_verts = get_fsaverage_medial_vertices()
# Load and morph labels
fslabels = mne.read_labels_from_annot('fsaverage',
                                      'aparc_sub', 'both',
                                      subjects_dir=defaults.subjects_dir)
fslabels = [label for label in fslabels
            if not label.name.startswith('unknown')]
label_nms = [rr.name for rr in fslabels]
picks = pd.read_csv(op.join(defaults.static, 'picks.tsv'), sep='\t')
picks.drop(picks[picks.id.isin(defaults.exclude)].index, inplace=True)
picks.sort_values(by='id', inplace=True)
for aix, age in enumerate(defaults.ages):
    subjects = ['genz%s' % ss for ss in picks[picks.ag == age].id]
    degrees = np.zeros((len(subjects), len(defaults.bands),
                        len(fslabels)))
    corr_mats = np.zeros((len(subjects), len(defaults.bands), len(fslabels),
                          len(fslabels)))
    for si, subject in enumerate(subjects):
        bem_dir = os.path.join(defaults.subjects_dir, subject, 'bem')
        bem_fname = os.path.join(bem_dir, '%s-5120-bem-sol.fif' % subject)
        src_fname = os.path.join(bem_dir, '%s-oct-6-src.fif' % subject)
        subj_dir = os.path.join(defaults.megdata, subject)
        raw_fname = os.path.join(subj_dir, 'sss_pca_fif',
                                 '%s_rest_01_allclean_fil100_raw_sss.fif' %
                                 subject)
        erm_fname = os.path.join(subj_dir, 'sss_pca_fif',
                                 '%s_erm_01_allclean_fil100_raw_sss.fif' %
                                 subject)
        trans_fname = os.path.join(subj_dir, 'trans',
                                   '%s-trans.fif' % subject)
        eps_dir = os.path.join(subj_dir, 'epochs')
        eps_fname = op.join(eps_dir, 'All_sss_%s-epo.fif' % subject)
        src_dir = os.path.join(subj_dir, 'source')
        # Load raw
        print('    Loading data for %s' % subject)
        raw = mne.io.read_raw_fif(raw_fname)
        if not raw.info['highpass'] == 0:
            print('%s acquisition HP greater than DC' % subject)
            continue
        if not os.path.exists(eps_dir):
            os.mkdir(eps_dir)
        if not os.path.exists(src_dir):
            os.mkdir(src_dir)
        raw.load_data().resample(new_sfreq, n_jobs=config.N_JOBS)
        raw_erm = mne.io.read_raw_fif(erm_fname)
        raw_erm.load_data().resample(new_sfreq, n_jobs=config.N_JOBS)
        raw_erm.add_proj(raw.info['projs'])
        # ERM covariance
        cov = compute_raw_covariance(raw_erm, n_jobs=defaults,
                                     rank='full', method='oas')

        # Make forward stack and get transformation matrix
        src = mne.read_source_spaces(src_fname)
        bem = mne.read_bem_solution(bem_fname)
        trans = mne.read_trans(trans_fname)
        fwd = mne.make_forward_solution(
            raw.info, trans, src=src, bem=bem, eeg=False, verbose=True)
        # epoch raw into 5 sec trials
        print('      \nLoading ...%s' % op.relpath(eps_fname, defaults.megdata))
        events = mne.make_fixed_length_events(raw, duration=5.)
        epochs = mne.Epochs(raw, events=events, tmin=0, tmax=5.,
                            baseline=None, reject=None, preload=True)
        if not op.isfile(eps_fname):
            # k-fold CV thresholded artifact rejection
            ar = AutoReject()
            epochs = ar.fit_transform(epochs)
            print('      \nSaving ...%s' % op.relpath(eps_fname,
                                                      defaults.megdata))
            epochs.save(eps_fname, overwrite=True)
        epochs = read_epochs(eps_fname)
        print('%d, %d (Epochs, drops)' %
              (len(events), len(events) - len(epochs.selection)))
        # fw = epochs.plot_psd()
        idx = np.setdiff1d(np.arange(len(events)), epochs.selection)
        for ix, (kk, vv) in enumerate(defaults.bands.items()):
            lf, hf = vv
            # r = raw.copy().filter(lf, hf, fir_window='blackman',
            #                       method='iir', n_jobs=config.N_JOBS)
            epochs_ = epochs.copy().filter(lf, hf, method='iir',
                                           fir_window='blackman',
                                           pad='constant_values',
                                           n_jobs=config.N_JOBS)
            mne.Info.normalize_proj(epochs_.info)
            # regularize covariance
            rank = compute_rank(cov, rank='full', info=epochs_.info)
            cov = regularize(cov, epochs_.info, rank=rank)
            inv = make_inverse_operator(epochs_.info, fwd, cov)
            # Compute label time series and do envelope correlation
            stcs = apply_inverse_epochs(epochs_, inv, lambda2=1. / 9.,
                                        pick_ori='normal',
                                        return_generator=True)
            labels = mne.morph_labels(fslabels, subject,
                                      subjects_dir=defaults.subjects_dir)
            label_ts = mne.extract_label_time_course(
                stcs, labels, fwd['src'], return_generator=True,
                verbose=True)
            corr_mats[si, ix] = envelope_correlation(label_ts)
            # Compute pairwise degree source connectivity amongst labels
            degrees[si, ix] = mne.connectivity.degree(corr_mats[si, ix], 0.15)
    for ix, (kk, vv) in enumerate(defaults.bands.items()):
        corr_ = corr_mats[:, ix].mean(0)
        # Plot group narrow-band corr matrix
        fig, ax = plt.subplots(figsize=(4, 4))
        img = ax.imshow(corr_, cmap='viridis',
                        clim=np.percentile(corr_, [5, 95]),
                        interpolation='nearest', origin='lower')
        fig.suptitle('%d - %dHz correlation matrix' % (vv[0], vv[1]))
        fig.colorbar(img, ax=ax)
        fig.tight_layout()
        fig.savefig(op.join(defaults.figs_dir, 'genz_%s_%s_corrmat.png'
                            % (age, kk)))
    # container for age x band x subj x roi connectivity data
    coords = {
        'id': subjects,
        'band': list(defaults.bands.keys()),
        'roi': label_nms
        }
    dsx = xr.Dataset({'degree': (['id', 'band', 'roi'], degrees)},
                     coords=coords)
    dsx.to_netcdf(op.join(defaults.datadir, 'genz_%s_degree.nc' % age))
    plt.close('all')
