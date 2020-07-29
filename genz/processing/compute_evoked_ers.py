#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compute evoked ERS envolope power"""

import os.path as op

import mne
from meeg_preprocessing import config
from mne import (
    compute_raw_covariance
    )
from mne.connectivity import envelope_correlation
from mne.filter import next_fast_len
import matplotlib.pyplot as plt
from genz import (
    defaults, funcs
    )

new_sfreq = defaults.new_sfreq
n_fft = next_fast_len(int(round(4 * new_sfreq)))
subject = 'genz529_17a'
work_dir = '/media/ktavabi/ALAYA/data/ilabs/sandbox/%s' % subject
# Read epochs
eps_fname = op.join(work_dir, 'epochs/All_80-sss_%s-epo.fif' % subject)

# Read forward operator and point to freesurfer subject directory
subjects_dir = op.join(defaults.subjects_dir)

fwd = mne.read_forward_solution(
    op.join(work_dir, 'forward/%s_aud-sss-fwd.fif' % subject))
# Load labels
fslabels = mne.read_labels_from_annot('fsaverage',
                                      'aparc_sub', 'both',
                                      subjects_dir=defaults.subjects_dir)
fslabels = [label for label in fslabels
            if not label.name.startswith('unknown')]
lp, hp = 29, 13
raw_fname = op.join(work_dir,
                    'sss_pca_fif/%s_erm_01_allclean_fil80_raw_sss.fif' %
                    subject)  # noqa
rest_fname = op.join(work_dir, 'sss_pca_fif',
                     '%s_rest_01_allclean_fil100_raw_sss.fif' %
                     subject)
erm_fname = op.join(work_dir, 'sss_pca_fif',
                    '%s_erm_01_allclean_fil100_raw_sss.fif' %
                    subject)

# Load raw
print('    Loading data for %s' % subject)
raw = mne.io.read_raw_fif(raw_fname)
raw.load_data().resample(new_sfreq, n_jobs=config.N_JOBS)
raw_erm = mne.io.read_raw_fif(erm_fname)
raw_erm.load_data().resample(new_sfreq, n_jobs=config.N_JOBS)
raw_erm.add_proj(raw.info['projs'])
raw_rest = mne.io.read_raw_fif(rest_fname)
raw_rest.load_data().resample(new_sfreq, n_jobs=config.N_JOBS)
raw_rest.add_proj(raw.info['projs'])
# Covariance
cov_erm = compute_raw_covariance(raw_erm, n_jobs=12,
                                 reject=dict(grad=4000e-13,
                                             mag=4e-12),
                                 flat=dict(grad=1e-13, mag=1e-15))
cov_rest = compute_raw_covariance(raw_rest, n_jobs=12,
                                  reject=dict(grad=4000e-13,
                                              mag=4e-12),
                                  flat=dict(grad=1e-13, mag=1e-15))
label_ts_erm = funcs.extract_labels_timeseries(subject, raw, hp, lp,
                                               cov_erm, fwd,
                                               defaults.subjects_dir,
                                               eps_fname, fslabels,
                                               return_generator=True)
label_ts_rest = funcs.extract_labels_timeseries(subject, raw, hp, lp,
                                                cov_rest, fwd,
                                                defaults.subjects_dir,
                                                eps_fname, fslsabels,
                                                return_generator=True)
names = ['']
corr_mats = [envelope_correlation(ls) for ls in [label_ts_erm, label_ts_rest]]
# Compute pairwise degree source connectivity amongst labels
degrees = [mne.connectivity.degree(ma) for ma in corr_mats]

# let's plot this matrix
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(corr_mats[1], cmap='viridis', clim=np.percentile(corr_mats[1], [5,
                                                                          95]))
fig.tight_layout()

threshold_prop = 0.15  # percentage of strongest edges to keep in the graph
bem_dir = op.join(defaults.subjects_dir, subject, 'bem')
src_fname = op.join(bem_dir, '%s-oct-6-src.fif' % subject)
src = mne.read_source_spaces(src_fname)
stcs = []
for cov, deg in zip(['ERM', 'Rest'], degrees):
    title = '%s source power in the 13-29 Hz frequency band' % cov
    stc = mne.labels_to_stc(fslabels, deg)
    stcs.append(stc.in_label(mne.Label(src[0]['vertno'], hemi='lh') +
                             mne.Label(src[1]['vertno'], hemi='rh')))
brain = stcs[1].plot(clim=dict(kind='percent', lims=[75, 85, 95]),
                     colormap='gnuplot',
                     subjects_dir=subjects_dir, views='dorsal', hemi='both',
                     smoothing_steps=25,
                     time_label='Rest covar - beta power')
