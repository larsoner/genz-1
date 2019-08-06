#!/usr/bin/env python

"""Plot functional connectivity computed on envolope power correlation."""

__author__ = 'Kambiz Tavabi'
__copyright__ = 'Copyright 2018, Seattle, Washington'
__credits__ = ['Eric Larson']
__license__ = 'MIT'
__version__ = '1.0.1'
__maintainer__ = 'Kambiz Tavabi'
__email__ = 'ktavabi@uw.edu'
__status__ = 'Development'

import os
import os.path as op

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import xarray as xr
from autoreject import AutoReject
from meeg_preprocessing import utils
from mne import (
    read_epochs, compute_raw_covariance,
    compute_rank
    )
from mne.connectivity import envelope_correlation
from mne.cov import regularize
from mne.externals.h5io import write_hdf5
from mne.filter import next_fast_len
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, \
    compute_source_psd_epochs
from mne.time_frequency import psd_multitaper
from mnefun import get_fsaverage_medial_vertices

sns.set(style='ticks')
utils.setup_mpl_rcparams(font_size=10)

datapath = '/mnt/jaba/meg/genz_resting'
subjects_dir = '/mnt/jaba/meg/genz/anatomy'
figs_dir = '/home/ktavabi/Github/genz/figures'
data_dir = '/home/ktavabi/Github/genz/data'
# define frequencies of interest
bandwidth = 'hann'  # bandwidth of the windows in Hz
bands = {
        'DC': (0.01, 2), 'delta': (2, 4), 'theta': (5, 7),
        'alpha': (8, 12), 'beta': (13, 29), 'gamma': (30, 50)
        }
ages = np.arange(9, 13, 2)
new_sfreq = 200.
n_fft = next_fast_len(int(round(4 * new_sfreq)))
lims = [75, 85, 95]
medial_verts = get_fsaverage_medial_vertices()
# Load and morph labels
fslabels = mne.read_labels_from_annot('fsaverage', 'aparc_sub', 'both',
                                      subjects_dir=subjects_dir)
fslabels = [label for label in fslabels
            if not label.name.startswith('unknown')]
label_nms = [rr.name for rr in fslabels]
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
for aix, age in enumerate(ages):
    subjects = ['genz%s' % ss for ss in picks[picks.ag == age].id[:2]]
    corr = np.zeros((len(bands), len(subjects), len(fslabels), len(fslabels)))
    degrees = np.zeros((len(bands), len(subjects), len(fslabels)))
    for ix, (kk, vv) in enumerate(bands.items()):
        lf, hf = vv
        psds = list()
        for si, subject in enumerate(subjects):
            bem_dir = os.path.join(subjects_dir, subject, 'bem')
            bem_fname = os.path.join(bem_dir, '%s-5120-bem-sol.fif' % subject)
            src_fname = os.path.join(bem_dir, '%s-oct-6-src.fif' % subject)
            subj_dir = os.path.join(datapath, subject)
            raw_fname = os.path.join(subj_dir, 'sss_pca_fif',
                                     '%s_rest_01_allclean_fil100_raw_sss.fif' %
                                     subject)
            erm_fname = os.path.join(subj_dir, 'sss_pca_fif',
                                     '%s_erm_01_allclean_fil100_raw_sss.fif' %
                                     subject)
            trans_fname = os.path.join(subj_dir, 'trans',
                                       '%s-trans.fif' % subject)
            eps_dir = os.path.join(subj_dir, 'epochs')
            eps_fname = op.join(eps_dir, 'All_%d-%d-sss_%s-epo.fif' % (lf, hf,
                                                                       subject))
            src_dir = os.path.join(subj_dir, 'source')
            # Load and preprocess the data
            print('    Loading data for %s' % subject)
            raw = mne.io.read_raw_fif(raw_fname)
            if not raw.info['highpass'] == 0:
                print('%s acquisition HP greater than DC' % subject)
                continue
            if not os.path.exists(eps_dir):
                os.mkdir(eps_dir)
            if not os.path.exists(src_dir):
                os.mkdir(src_dir)
            raw.load_data().resample(new_sfreq, n_jobs='cuda')
            
            psd_avg_ = 0.
            for i, stc in enumerate(psd):
                psd_avg_ += stc.data
            psd_avg_ /= len(epochs)
            freqs = stc.times  # the frequencies are stored here
            psds.append(psd_avg_.mean(axis=0))
            raw_erm = mne.io.read_raw_fif(erm_fname)
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
                print('      \nSaving %s-band epochs for %s' % (kk, subject))
                epochs.save(eps_fname, overwrite=True)
            epochs = read_epochs(eps_fname)
            print('%d, %d (Epochs, drops)' %
                  (len(epochs), len(epochs.drop_log) - len(epochs)))
            mne.Info.normalize_proj(epochs.info)
            # covariance
            cov = compute_raw_covariance(raw_erm, n_jobs='cuda',
                                         rank='full', method='oas')
            rank = compute_rank(cov, rank='full', info=epochs.info)
            cov = regularize(cov, epochs.info, rank=rank)
            # Make forward stack and get transformation matrix
            src = mne.read_source_spaces(src_fname)
            bem = mne.read_bem_solution(bem_fname)
            trans = mne.read_trans(trans_fname)
            fwd = mne.make_forward_solution(
                    raw.info, trans, src=src, bem=bem, eeg=False, verbose=True)
            inv = make_inverse_operator(epochs.info, fwd, cov)
            # Compute label time series and do envelope correlation
            stcs = apply_inverse_epochs(epochs, inv, lambda2=1. / 9.,
                                        pick_ori='normal',
                                        return_generator=True)
            labels = mne.morph_labels(fslabels, subject,
                                      subjects_dir=subjects_dir)
            label_ts = mne.extract_label_time_course(
                    stcs, labels, fwd['src'], return_generator=True,
                    verbose=True)
            corr[ix, si] = envelope_correlation(label_ts)
            # Compute the degree
            degrees[ix, si] = mne.connectivity.degree(corr[ix, si], 0.15)
        # Plot group averaged corr matrix
        corr_avg = corr[ix].mean(axis=0)
        fig, ax = plt.subplots(figsize=(4, 4))
        img = ax.imshow(corr_avg, cmap='viridis',
                        clim=np.percentile(corr, [5, 95]),
                        interpolation='nearest', origin='lower')
        fig.suptitle('%d - %dHz correlation matrix' % (lf, hf))
        fig.colorbar(img, ax=ax)
        fig.tight_layout()
        fig.savefig(op.join(figs_dir, 'genz_%s_%s_corrmat.png'
                            % (age, kk)))
        # plot group average band-psd
        psd_avg = np.asarray(psds).mean(0)
        psd_sem = stats.sem(np.asarray(psds), axis=0)
        fig, ax = plt.subplots()
        ax.plot(freqs, psd_avg, label='Mean')
        ax.fill_between(freqs, psd_avg + psd_sem, psd_avg - psd_sem,
                        facecolor='cyan', alpha=0.5, label='SEM')
        ax.set_xlabel('Freq (Hz)')
        ax.set_xlim(freqs[[0, -1]])
        ax.set_ylabel('AU')
        ax.set_title('Grand average %s (%d-%dHz) PSD' % (kk, lf, hf))
        fig.tight_layout()
        fig.savefig(op.join(figs_dir, 'genz_%s_%s_psd.png'
                            % (age, kk)))
        # plot group degree on fsaverage
        stc = mne.labels_to_stc(fslabels,
                                mne.connectivity.degree(corr_avg, 0.15))
        stc = stc.in_label(mne.Label(np.arange(10242), hemi='lh') +
                           mne.Label(np.arange(10242), hemi='rh'))
        # lims = len(fslabels) * np.array([0.2, 0.4, 0.6])
        brain = stc.plot(subject='fsaverage', hemi='split', colormap='gnuplot',
                         views=['lat', 'med'],
                         clim=dict(kind='percent', lims=lims),
                         surface='inflated', subjects_dir=subjects_dir,
                         smoothing_steps=25, time_label='%s band' % kk)
        brain.save_image(op.join(figs_dir,
                                 'genz_%s_%s_degrees.png' % (age, kk)))
    # container for age x band x subj x roi connectivity data
    dsx = xr.DataArray(degrees, coords=[list(bands.keys()),
                                        subjects,
                                        label_nms],
                       dims=['band', 'subject', 'roi'])
    write_hdf5(op.join(data_dir, 'genz_%s_degree.csv' % age), degrees,
               overwrite=True)
