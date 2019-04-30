

import os

import numpy as np
import matplotlib.pyplot as plt

from mne.filter import next_fast_len
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.connectivity import envelope_correlation
from mayavi import mlab
import mne
from autoreject import AutoReject

datapath = '/Users/ktavabi/Sandbox'
subject = 'genz105_9a'


subjects_dir = os.path.join(datapath, 'Anatomy')
bem_dir = os.path.join(subjects_dir, subject, 'bem')
bem_fname = os.path.join(bem_dir, '%s-5120-bem-sol.fif' % subject)
fwd_fname = os.path.join(bem_dir, '')
src_fname = os.path.join(bem_dir, '%s-oct-6-src.fif' % subject)
raw_fname = os.path.join(datapath,
                         '%s/sss_pca_fif/%s_rest_01_allclean_fil100_raw_sss'
                         '.fif' %
                         (subject, subject))
raw_erm_fname = os.path.join(datapath,
                             '%s/sss_pca_fif/%s_erm_01_allclean_fil100_raw_sss'
                             '.fif' % (subject, subject))
trans_fname = os.path.join(datapath, '%s/trans/%s-trans.fif' % (subject,
                                                                subject))

# Load data, resample
new_sfreq = 200.
n_fft = next_fast_len(int(round(4 * new_sfreq)))
raw = mne.io.read_raw_fif(raw_fname)
raw.load_data().resample(new_sfreq, n_jobs=4)
raw_erm = mne.io.read_raw_fif(raw_erm_fname)
raw_erm.load_data().resample(new_sfreq, n_jobs=4)
raw_erm.add_proj(raw.info['projs'])
raw.filter(14, 30, n_jobs=4)
# make epochs

events = mne.make_fixed_length_events(raw, duration=5.)
epochs = mne.Epochs(raw, events=events, tmin=0, tmax=5.,
                    baseline=None, reject=None, preload=True)
ar = AutoReject()
epochs = ar.fit_transform(epochs)
# cov
cov = mne.compute_raw_covariance(raw_erm, n_jobs=4)

# Make forward stack and get transformation matrix
src = mne.read_source_spaces(src_fname)
bem = mne.read_bem_solution(bem_fname)
trans = mne.read_trans(trans_fname)
# compute fwd & inverse
fwd = mne.make_forward_solution(
    raw.info, trans, src=src, bem=bem, eeg=False, verbose=True)
inv = make_inverse_operator(epochs.info, fwd, cov)

# check alignment
fig = mne.viz.plot_alignment(
    raw.info, trans=trans, subject=subject, subjects_dir=subjects_dir,
    dig=True, coord_frame='meg')
mlab.view(0, 90, focalpoint=(0., 0., 0.), distance=0.6, figure=fig)

# Compute label time series and do envelope correlation
labels = mne.read_labels_from_annot(subject, 'aparc.a2009s',
                                    subjects_dir=subjects_dir)
stcs = apply_inverse_epochs(epochs, inv, lambda2=1. / 9., pick_ori='normal')
label_ts = mne.extract_label_time_course(
    stcs, labels, inv['src'], return_generator=True)
corr = envelope_correlation(label_ts)

# let's plot this matrix
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(corr, cmap='viridis', clim=np.percentile(corr, [5, 95]))
fig.tight_layout()

# Compute the degree and plot it
degree = mne.connectivity.degree(corr, 0.15)
stc = mne.labels_to_stc(labels, degree)
stc = stc.in_label(mne.Label(inv['src'][0]['vertno'], hemi='lh') +
                   mne.Label(inv['src'][1]['vertno'], hemi='rh'))
brain = stc.plot(
    clim=dict(kind='percent', lims=[75, 85, 95]), colormap='gnuplot',
    subjects_dir=subjects_dir, views='dorsal', hemi='both',
    smoothing_steps=25, time_label='Beta band')
brain.show_view(dict(azimuth=0, elevation=0), roll=0)
