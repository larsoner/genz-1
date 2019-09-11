#!/usr/bin/env python

"""plot_psd_raw.py: plot raw power spectral density."""

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
import pandas as pd
import seaborn as sns
from autoreject import AutoReject
from meeg_preprocessing import config
from meeg_preprocessing import utils
from mne.filter import next_fast_len
from mne.time_frequency import psd_multitaper
from scipy import stats

from genz import defaults

sns.set(style='ticks')
utils.setup_mpl_rcparams(font_size=10)

datapath = defaults.megdata
subjects_dir = defaults.subjects_dir
figs_dir = defaults.figs_dir
data_dir = defaults.datadir
static = defaults.static

picks = pd.read_csv(op.join(static, 'picks.tsv'), sep='\t')
picks.drop(picks[picks.id.isin(defaults.exclude)].index, inplace=True)
picks.sort_values(by='id', inplace=True)
new_sfreq = 200.
n_fft = next_fast_len(int(round(4 * new_sfreq)))
kinds = ['mag', 'grad']
ages = np.arange(9, 19, 2)
exclude = list()
for aix, age in enumerate(ages):
    subjects = ['genz%s' % ss for ss in picks[picks.ag == age].id]
    psds = list()
    for si, subject in enumerate(subjects):
        subj_dir = os.path.join(datapath, subject)
        raw_fname = os.path.join(subj_dir, 'sss_pca_fif',
                                 '%s_rest_01_allclean_fil100_raw_sss.fif' %
                                 subject)
        erm_fname = os.path.join(subj_dir, 'sss_pca_fif',
                                 '%s_erm_01_allclean_fil100_raw_sss.fif' %
                                 subject)
        # Load raw and epoch
        print('    Loading data for %s' % subject)
        raw = mne.io.read_raw_fif(raw_fname)
        if not raw.info['highpass'] == 0:
            exclude.append(subject)
            continue
        raw.load_data().resample(new_sfreq, n_jobs='cuda')
        events = mne.make_fixed_length_events(raw, duration=5.)
        epochs = mne.Epochs(raw, events=events, tmin=0, tmax=5.,
                            baseline=None, reject=None, preload=True)
        # k-folc CV thresholded artifact rejection
        ar = AutoReject()
        epochs = ar.fit_transform(epochs)
        # Compute raw PSD using multitaper windows
        for kk, kind in enumerate(kinds):
            eps = epochs.copy().pick_types(meg=kind)
            psd, fqs = psd_multitaper(eps, fmax=80, adaptive=True,
                                      proj=True, verbose=True,
                                      n_jobs=config.N_JOBS)
            psd_eps = psd.mean(0)  # average over epochs
            if kk == 0:
                psds_ = np.zeros((len(kinds), len(fqs)))
            psds_[kk] = psd_eps.mean(0)  # average over channels
        psds.append(psds_)
    # Convert to dB
    psds_db = np.log10(np.asarray(psds) / 32768)  # 32768 for int16
    psds_mean = psds_db.mean(0)  # average over subjects
    psds_sem = stats.sem(psds_db, axis=0)
    # plot group average psd
    fig, axs = plt.subplots(len(kinds), 1, sharex=True)
    titles = ['Magnetometers', 'Gradiometers']
    for ii, ax in enumerate(axs):
        ax.plot(fqs.T, psds_mean[ii])
        ax.fill_between(fqs.T, psds_mean[ii] + psds_sem[ii],
                        psds_mean[ii] - psds_sem[ii],
                        facecolor='turquoise', alpha=0.9)
        ax.set_title('%s PSD' % titles[ii])
        if ii == 0:
            ax.set_ylabel(r'$fT^2/Hz (db)$')
        elif ii == 1:
            ax.set_xlabel('Freq (Hz)')
            ax.set_xlim(fqs[[0, -1]])
            ax.set_ylabel(r'$fT/cm^2/Hz (db)$')
            ax.legend(['Mean', 'SEM'],
                      loc=7, bbox_to_anchor=(1, .9),
                      numpoints=1, frameon=False)
    fig.tight_layout()
    fig.savefig(op.join(figs_dir, 'genz_%s_psd.png' % age))
print('Subjects with acquisition HP greater than zero:\n    ' % exclude)
