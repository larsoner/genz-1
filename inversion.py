# -*- coding: utf-8 -*-

""" """

# Author(s): Kambiz Tavabi <ktavabi@gmail.com>

import os.path as op
import numpy as np
import mne
from mne import EvokedArray
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.inverse_sparse.mxne_inverse import _prepare_gain

# Set parameters
data_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample')
subjects_dir = op.join(data_path, 'subjects')
raw_fname = op.join(data_path, 'sample_audvis_filt-0-40_raw.fif')
event_fname = op.join(data_path, 'sample_audvis_filt-0-40_raw-eve.fif')
fname_evoked = op.join(data_path, 'sample_audvis-ave.fif')
fname_trans = op.join(data_path, 'sample_audvis_raw-trans.fif')
fname_src = op.join(mne.datasets.sample.data_path(),
                    'subjects', 'sample', 'bem', 'sample-oct-6-src.fif')
fname_bem = op.join(mne.datasets.sample.data_path(),
                    'subjects', 'sample',
                    'bem', 'sample-5120-5120-5120-bem-sol.fif')

# Read raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, 30, fir_design='firwin')
events = mne.read_events(event_fname)

# Read epochs
event_id = dict(aud_r=1)
tmin, tmax = -0.2, 0.5
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    baseline=(None, 0),
                    reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
epochs.load_data().pick_types(meg=True)
noise_cov = mne.compute_covariance(
    epochs, tmax=0., method=['shrunk', 'empirical'])
fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw.info)

# Compute the evoked response
evoked = epochs.average()
evoked.plot()
evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type='mag')
evoked.plot_white(noise_cov)  # Show whitening
all_ch_names = evoked.ch_names

# Read the forward solution and compute the inverse operator
info = mne.io.read_info(fname_evoked)
trans = mne.read_trans(fname_trans)
src = mne.read_source_spaces(fname_src)
bem = mne.read_bem_solution(fname_bem)
fwd = mne.make_forward_solution(info, trans, src, bem, meg=True,
                                eeg=False,
                                mindist=0.0, ignore_ref=False, n_jobs=1,
                                verbose=None)
fwd = mne.convert_forward_solution(fwd, surf_ori=True)

# Inversion
inverse = make_inverse_operator(info, fwd, noise_cov, loose=.2)
stc_ = apply_inverse(EvokedArray(np.eye(len(info['ch_names'])), info),
                     inverse).data
stc_.plot(subject='sample', hemi='both', subjects_dir=subjects_dir,
          initial_time=0.1)
stc = apply_inverse(evoked, info, inverse)
stc.plot(subject='sample', hemi='both', subjects_dir=subjects_dir,
         initial_time=0.1)
assert np.allclose(stc.data, np.dot(stc_, evoked.data))


