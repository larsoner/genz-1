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

import mne
import numpy as np
import scipy.io as sio
import pandas as pd
from autoreject import AutoReject
from meeg_preprocessing import config
from mne import (
    read_epochs, compute_raw_covariance,
    compute_rank
    )
from mne.cov import regularize
from mne.filter import next_fast_len
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from surfer import Brain
import matplotlib.pyplot as plt
from genz import defaults


def extract_labels_timeseries(subj, r, highpass, lowpass, covar, labels,
                              subjects_dir,
                              return_generator=False):
    # epoch raw into 5 sec trials
    events = mne.make_fixed_length_events(r, duration=5.)
    epochs = mne.Epochs(r, events=events, tmin=0, tmax=5.,
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
    # r = r.copy().filter(lf, hf, fir_window='blackman',
    #                       method='iir', n_jobs=config.N_JOBS)
    epochs_ = epochs.copy().filter(highpass, lowpass, method='iir',
                                   fir_window='blackman',
                                   pad='constant_values', n_jobs=config.N_JOBS)
    mne.Info.normalize_proj(epochs_.info)
    # regularize covariance
    rank = compute_rank(covar, rank='full', info=epochs_.info)
    covar = regularize(covar, epochs_.info, rank=rank)
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


new_sfreq = 200
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
        lp, hp = vv
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
            ts_ = extract_labels_timeseries(subject, raw, lp, hp, cov,
                                            [wernicke_roi, broca_roi],
                                            defaults.subjects_dir)
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


