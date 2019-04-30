"""
================================
Brainstorm resting state dataset
================================

Here we compute the resting state from raw for the
Brainstorm tutorial dataset, see [1]_.

The pipeline is meant to mirror the Brainstorm
`resting tutorial pipeline <bst_tut_>`_. The steps we use are:

1. Filtering: downsample heavily.
2. Artifact detection: use SSP for EOG and ECG.
3. Source localization: dSPM, depth weighting, cortically constrained.
4. Frequency: power spectrum density (Welch), 4 sec window, 50% overlap.
5. Standardize: normalize by relative power for each source.

References
----------
.. [1] Tadel F, Baillet S, Mosher JC, Pantazis D, Leahy RM.
       Brainstorm: A User-Friendly Application for MEG/EEG Analysis.
       Computational Intelligence and Neuroscience, vol. 2011, Article ID
       879716, 13 pages, 2011. doi:10.1155/2011/879716

.. _bst_tut: https://neuroimage.usc.edu/brainstorm/Tutorials/RestingOmega
"""
# sphinx_gallery_thumbnail_number = 3

# Authors:
#

import os
from mne.filter import next_fast_len
from mayavi import mlab
import mne


def plot_band(band):
    title = "%s (%d-%d Hz)" % ((band.capitalize(),) + freq_bands[band])
    topos[band].plot_topomap(
        times=0., scalings=1., cbar_fmt='%0.1f', vmin=0, cmap='inferno',
        time_format=title)
    brain = stcs[band].plot(
        subject=subject, subjects_dir=subjects_dir, views='cau', hemi='both',
        time_label=title, title=title, smoothing_steps=25,
        clim=dict(kind='percent', lims=[75, 85, 95]), colormap='gnuplot')
    brain.show_view(dict(azimuth=0, elevation=0), roll=0)
    return fig, brain


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


# Make forward stack and get transformation matrix
src = mne.read_source_spaces(src_fname)
bem = mne.read_bem_solution(bem_fname)
trans = mne.read_trans(trans_fname)

# check alignment
fig = mne.viz.plot_alignment(
    raw.info, trans=trans, subject=subject, subjects_dir=subjects_dir,
    dig=True, coord_frame='meg')
mlab.view(0, 90, focalpoint=(0., 0., 0.), distance=0.6, figure=fig)
fwd = mne.make_forward_solution(
    raw.info, trans, src=src, bem=bem, eeg=False, verbose=True)

##############################################################################
# Compute and apply inverse to PSD estimated using multitaper + Welch

noise_cov = mne.compute_raw_covariance(raw_erm, n_jobs=4)

inverse_operator = mne.minimum_norm.make_inverse_operator(
    raw.info, forward=fwd, noise_cov=noise_cov, verbose=True)

stc_psd, evoked_psd = mne.minimum_norm.compute_source_psd(
    raw, inverse_operator, lambda2=1. / 9., method='MNE', n_fft=n_fft,
    dB=False, return_sensor=True, n_jobs=4, verbose=True)

##############################################################################
# Group into frequency bands, then normalize each source point and sensor
# independently. This makes the value of each sensor point and source location
# in each frequency band the percentage of the PSD accounted for by that band.

freq_bands = dict(beta=(14, 30))
topos = dict()
stcs = dict()
topo_norm = evoked_psd.data.sum(axis=1, keepdims=True)
stc_norm = stc_psd.sum()
# Normalize each source point by the total power across freqs
for band, limits in freq_bands.items():
    data = evoked_psd.copy().crop(*limits).data.sum(axis=1, keepdims=True)
    topos[band] = mne.EvokedArray(100 * data / topo_norm, evoked_psd.info)
    stcs[band] = 100 * stc_psd.copy().crop(*limits).sum() / stc_norm.data

for freqs in ['beta']:
    fig, brain = plot_band(freqs)
