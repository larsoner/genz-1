#!/usr/bin/env python

"""compute_psd_connectivity.py: compute functional connectivity from spectral
power.
    Description: 1. procedure
"""

__author__ = "Kambiz Tavabi"
__copyright__ = "Copyright 2019, Seattle, Washington"
__license__ = "MIT"
__version__ = "0.2"
__maintainer__ = "Kambiz Tavabi"
__email__ = "ktavabi@uw.edu"

import os
import os.path as op

import mne
import pandas as pd
from meeg_preprocessing import config
from mne import (
    compute_raw_covariance
    )
from mne.filter import next_fast_len
from mne.minimum_norm import make_inverse_operator, compute_source_psd

from genz import defaults

picks = pd.read_csv(op.join(defaults.static, 'picks.tsv'), sep='\t')
picks.drop(picks[picks.id.isin(defaults.exclude)].index, inplace=True)
picks.sort_values(by='id', inplace=True)
new_sfreq = 500.
n_fft = next_fast_len(int(round(4 * new_sfreq)))
for aix, age in enumerate(defaults.ages):
    stc_psds = []
    evoked_psds = []
    subjects = ['genz%s' % ss for ss in picks[picks.ag == age].id]
    for si, subject in enumerate(subjects):
        subj_dir = os.path.join(defaults.megdata, subject)
        bem_dir = os.path.join(defaults.subjects_dir, subject, 'bem')
        bem_fname = os.path.join(bem_dir, '%s-5120-bem-sol.fif' % subject)
        src_fname = os.path.join(bem_dir, '%s-oct-6-src.fif' % subject)
        trans_fname = os.path.join(subj_dir, 'trans',
                                   '%s-trans.fif' % subject)
        raw_fname = os.path.join(subj_dir, 'sss_pca_fif',
                                 '%s_rest_01_allclean_fil100_raw_sss.fif' %
                                 subject)
        erm_fname = os.path.join(subj_dir, 'sss_pca_fif',
                                 '%s_erm_01_allclean_fil100_raw_sss.fif' %
                                 subject)
        # Load raw
        print('    Loading data for %s' % subject)
        raw = mne.io.read_raw_fif(raw_fname)
        if not raw.info['highpass'] == 0:
            print('%s acquisition HP greater than DC' % subject)
            continue
        raw.load_data().resample(new_sfreq, n_jobs='cuda')
        raw_erm = mne.io.read_raw_fif(erm_fname)
        raw_erm.load_data().resample(new_sfreq, n_jobs='cuda')
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
        # Compute and apply inverse to PSD estimated using multitaper + Welch
        noise_cov = mne.compute_raw_covariance(raw_erm, n_jobs=18)
        inverse_operator = make_inverse_operator(raw.info, forward=fwd,
                                                 noise_cov=noise_cov,
                                                 verbose=True)
        stc_psd, evoked_psd = compute_source_psd(raw, inverse_operator,
                                                 method='MNE', dB=False,
                                                 return_sensor=True,
                                                 low_bias=True,
                                                 adaptive=True,
                                                 n_jobs=config.N_JOBS,
                                                 n_fft=n_fft,
                                                 verbose=True)
        stc_psds.append(stc_psd)
        evoked_psds.append(evoked_psd)
    grand_ave_stc /= len(stc_psds)
    grand_ave_evo = mne.grand_average(evoked_psds)
    topos = dict()
    stcs = dict()
    topo_norm = grand_ave_evo.data.sum(axis=1, keepdims=True)
    stc_norm = grand_ave_stc.sum()
    # Normalize each sensor/source by the total power across freqs
    for band, limits in defaults.bands.items():
        data = evoked_psd.copy().crop(*limits).data.sum(axis=1, keepdims=True)
        topos[band] = mne.EvokedArray(100 * data / topo_norm, evoked_psd.info)
        stcs[band] = 100 * stc_psd.copy().crop(*limits).sum() / stc_norm.data
