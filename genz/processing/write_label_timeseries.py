#!/usr/bin/env python

"""foobar.py does.
    Description:
        1. procedure
"""

__author__ = "Kambiz Tavabi"
__copyright__ = "Copyright 2019, Seattle, Washington"
__license__ = "MIT"
__version__ = "0.2"
__maintainer__ = "Kambiz Tavabi"
__email__ = "ktavabi@uw.edu"

import os
import os.path as op

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy.io as sio
from meeg_preprocessing import config
from mne import (
    compute_raw_covariance
    )
from mne.filter import next_fast_len
from surfer import Brain

from genz import (
    defaults, funcs
    )

new_sfreq = defaults.new_sfreq
n_fft = next_fast_len(int(round(4 * new_sfreq)))
# Load labels
fslabels = mne.read_labels_from_annot('fsaverage',
                                      'aparc_sub', 'both', surf_name='white',
                                      subjects_dir=defaults.subjects_dir)
wernicke_mni = np.array([-54, -47.5, 7.5]) / 1000
dists = [np.min(np.linalg.norm(l.pos - wernicke_mni, axis=-1)) for l in
         fslabels]
print([fslabels[idx] for idx in np.argsort(dists)[:10]])
wernicke_roi = np.array([fslabels[idx] for idx in np.argsort(dists)[:4]]).sum()
broca_mni = np.array([-51, 21.7, 7.5]) / 1000
dists = [np.min(np.linalg.norm(l.pos - broca_mni, axis=-1)) for l in fslabels]
print([fslabels[idx] for idx in np.argsort(dists)[:10]])
broca_roi = np.array([fslabels[idx] for idx in np.argsort(dists)[:6]]).sum()
brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=defaults.subjects_dir,
              cortex='low_contrast', background='white', size=(800, 600))
brain.add_label(wernicke_roi + broca_roi, borders=False)

picks = pd.read_csv(op.join(defaults.static, 'picks.tsv'), sep='\t')
picks.drop(picks[picks.id.isin(defaults.exclude)].index, inplace=True)
picks.sort_values(by='id', inplace=True)
for aix, age in enumerate(defaults.ages):
    subjects = ['genz%s' % ss for ss in picks[picks.ag == age].id]
    for ix, (kk, vv) in enumerate(defaults.bands.items()):
        hp, lp = vv
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
            print('      \nLoading ...%s' % op.relpath(eps_fname,
                                                       defaults.megdata))
            ts_ = funcs.extract_labels_timeseries(subject, raw, hp, lp, cov,
                                                  fwd, defaults.subjects_dir,
                                                  eps_fname,
                                                  [wernicke_roi, broca_roi],
                                                  return_generator=False)
            ts_ = np.array(ts_).mean(axis=0)
            mdict_ = {
                'wernicke': ts_[0],
                'broca': ts_[1]
                }
            fout_ = op.join(eps_dir, '%s_%s_label-ts.mat' % (subject, kk))
            sio.savemat(fout_, mdict_)
            if si == 0:
                ts = np.zeros((len(subjects), ts_.shape[0], ts_.shape[1]))
            ts[si] = ts_
        mdict = {
            'wernicke': ts.mean(axis=0)[0],
            'broca': ts.mean(axis=0)[1]
            }
        fout = op.join(defaults.datadir,
                       'genz_%d-%s_label-ts.mat' % (age, kk))
        sio.savemat(fout, mdict)
        # View source activations
        fig, ax = plt.subplots(1)
        tsec = np.linspace(0, 5, num=ts.shape[-1])
        ax.plot(tsec, ts.mean(axis=0)[0].T, linewidth=1, label='Wernicke')
        ax.plot(tsec, ts.mean(axis=0)[1].T, linewidth=1, label='Broca')
        ax.legend(loc='upper right')
        ax.set(xlabel='Time (sec)', ylabel='Source amplitude',
               title='ROI %s activity' % kk)
