#!/usr/bin/env python

"""Plot envolope power correlation connectivity."""

__author__ = "Kambiz Tavabi"
__copyright__ = "Copyright 2018, Seattle, Washington"
__credits__ = ["Goedel", "Escher", "Bach"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Kambiz Tavabi"
__email__ = "ktavabi@uw.edu"
__status__ = "Development"

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mne.filter import next_fast_len
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.connectivity import envelope_correlation
import mne
from autoreject import AutoReject
from mnefun import get_fsaverage_medial_vertices

datapath = '/mnt/jaba/meg/genz_resting'
subjects_dir = '/mnt/jaba/meg/genz/anatomy'
figs_dir = '/home/ktavabi/Github/genz/figures'
bands = [(14, 30)]  # beta
age = 17
new_sfreq = 200.
lims = [75, 85, 95]
medial_verts = get_fsaverage_medial_vertices()
fslabels = mne.read_labels_from_annot('fsaverage', 'aparc_sub', 'both',
                                      subjects_dir=subjects_dir)
fslabels = [label for label in fslabels
            if not label.name.startswith('unknown')]
picks = pd.read_csv('/home/ktavabi/Github/genz/static/picks.tsv', sep='\t')
exclude = ['104_9a',  # Too few EOG events
           '108_9a',  # Fix
           '113_9a',  # Too few ECG events
           '115_9a',  # no cHPI
           '121_9a',  # Source space missing points
           '209_11a',  # Too few EOG events
           '231_11a',  # twa_hp calc fail with assertion error
           '432_15a',  # Too few ECG events
           '510_17a',  # Too few ECG events
           '527_17a']  # Too few EOG events
picks.drop(picks[picks.id.isin(exclude)].index, inplace=True)
picks.sort_values(by='id', inplace=True)
subjects = picks[picks.ag == age].id
corr = np.zeros((len(subjects), len(fslabels), len(fslabels)))
degree = np.zeros((len(subjects), corr.shape[1]))
for band, name in zip(bands, ['beta']):
    lf, hf = band
    for si, subject in enumerate(subjects.values):
        subject = 'genz' + subject
        bem_dir = os.path.join(subjects_dir, subject, 'bem')
        bem_fname = os.path.join(bem_dir, '%s-5120-bem-sol.fif' % subject)
        src_fname = os.path.join(bem_dir, '%s-oct-6-src.fif' % subject)
        subj_dir = os.path.join(datapath, subject)
        raw_fname = os.path.join(subj_dir, 'sss_pca_fif',
                                 '%s_rest_01_allclean_fil100_raw_sss.fif' %
                                 subject)
        raw_erm_fname = os.path.join(subj_dir, 'sss_pca_fif',
                                     '%s_erm_01_allclean_fil100_raw_sss.fif' %
                                     subject)
        trans_fname = os.path.join(subj_dir, 'trans',
                                   '%s-trans.fif' % subject)
        eps_dir = os.path.join(subj_dir, 'epochs')
        eps_fname = op.join(eps_dir, 'All_%d-%d-sss_%s-epo.fif' % (lf, hf,
                                                                   subject))
        if not os.path.exists(eps_dir):
            os.mkdir(eps_dir)
        src_dir = os.path.join(subj_dir, 'source')
        if not os.path.exists(src_dir):
            os.mkdir(src_dir)
        # Load data, resample
        print('    Loading data for %s' % subject)
        n_fft = next_fast_len(int(round(4 * new_sfreq)))
        raw = mne.io.read_raw_fif(raw_fname)
        raw.load_data().resample(new_sfreq, n_jobs='cuda')
        raw_erm = mne.io.read_raw_fif(raw_erm_fname)
        raw_erm.load_data().resample(new_sfreq, n_jobs='cuda')
        raw_erm.add_proj(raw.info['projs'])
        # make epochs
        if not op.isfile(eps_fname):
            raw.filter(lf, hf, n_jobs='cuda')
            events = mne.make_fixed_length_events(raw, duration=5.)
            epochs = mne.Epochs(raw, events=events, tmin=0, tmax=5.,
                                baseline=None, reject=None, preload=True)
            # drop bad epochs
            ar = AutoReject()
            epochs = ar.fit_transform(epochs)
            print('      Saving %s-band epochs for %s' % (name, subject))
            epochs.save(eps_fname, overwrite=True)
        epochs = mne.read_epochs(eps_fname)
        mne.Info.normalize_proj(epochs.info)
        # covariance
        cov = mne.compute_raw_covariance(raw_erm, n_jobs='cuda')
        # Make forward stack and get transformation matrix
        src = mne.read_source_spaces(src_fname)
        bem = mne.read_bem_solution(bem_fname)
        trans = mne.read_trans(trans_fname)
        # compute fwd & inverse
        fwd = mne.make_forward_solution(
                raw.info, trans, src=src, bem=bem, eeg=False, verbose=True)
        inv = make_inverse_operator(epochs.info, fwd, cov)
        # Compute label time series and do envelope correlation
        stcs = apply_inverse_epochs(epochs, inv, lambda2=1. / 9.,
                                    pick_ori='normal')
        labels = mne.morph_labels(fslabels, subject,
                                  subjects_dir=subjects_dir)
        label_ts = mne.extract_label_time_course(
            stcs, labels, fwd['src'], return_generator=True)
        corr[si] = envelope_correlation(label_ts)
        # Compute the degree
        degree[si] = mne.connectivity.degree(corr[si], 0.15)
    # Plot group averaged corr matrix
    corr_avg = corr.mean(axis=0)
    fig, ax = plt.subplots(figsize=(4, 4))
    img = ax.imshow(corr_avg, cmap='viridis', clim=np.percentile(corr, [5, 95]),
                    interpolation='nearest', origin='lower')
    fig.suptitle('%d - %dHz correlation matrix' % (bands[0][0], bands[0][1]))
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    fig.savefig(op.join(figs_dir, '%s_%d-%d_envcorr_mat.png' % (age, lf, hf)))
    # plot group degree on fsaverage
    stc = mne.labels_to_stc(fslabels, mne.connectivity.degree(corr_avg, 0.15))
    stc = stc.in_label(mne.Label(np.arange(10242), hemi='lh') +
                       mne.Label(np.arange(10242), hemi='rh'))
    # lims = len(fslabels) * np.array([0.2, 0.4, 0.6])
    brain = stc.plot(subject='fsaverage', hemi='split', colormap='gnuplot',
                     views=['lat', 'med'],
                     clim=dict(kind='percent', lims=lims),
                     surface='inflated', subjects_dir=subjects_dir,
                     smoothing_steps=25, time_label='%s band' % name)
    brain.save_image(op.join(figs_dir,
                             '%s_%d-%d_envcorr_im.png' % (age, lf, hf)))
