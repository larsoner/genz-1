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
        time_label=title, title=title, colormap='inferno', smoothing_steps=25,
        clim=dict(kind='value', lims=(70, 85, 99)))
    brain.show_view(dict(azimuth=0, elevation=0), roll=0)
    return fig, brain

subject = 'genz530_17a'


subjects_dir = os.path.join(os.getcwd(), 'data/Anatomy')
bem_dir = os.path.join(subjects_dir, subject, 'bem')
bem_fname = os.path.join(bem_dir, '%s-5120-bem-sol.fif' % subject)
src_fname = os.path.join(bem_dir, '%s-oct-6-src.fif' % subject)
raw_fname = 'data/%s_rest_01_allclean_fil80_raw_sss.fif' % subject
raw_erm_fname = 'data/%s_erm_01_allclean_fil80_raw_sss.fif' % subject
trans_fname = 'data/%s-trans.fif' % subject

# Load data, resample
new_sfreq = 200.
n_fft = next_fast_len(int(round(4 * new_sfreq)))
raw = mne.io.read_raw_fif(raw_fname)
raw.load_data().resample(new_sfreq, n_jobs='cuda')
raw_erm = mne.io.read_raw_fif(raw_erm_fname)
raw_erm.load_data().resample(new_sfreq, n_jobs='cuda')⁄
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

noise_cov = mne.compute_raw_covariance(raw_erm, n_jobs=18)

inverse_operator = mne.minimum_norm.make_inverse_operator(
    raw.info, forward=fwd, noise_cov=noise_cov, verbose=True)

stc_psd, evoked_psd = mne.minimum_norm.compute_source_psd(
    raw, inverse_operator, lambda2=1. / 9., method='MNE', n_fft=n_fft,
    dB=False, return_sensor=True, n_jobs=18, verbose=True)

##############################################################################
# Group into frequency bands, then normalize each source point and sensor
# independently. This makes the value of each sensor point and source location
# in each frequency band the percentage of the PSD accounted for by that band.

freq_bands = dict(ultra=(0, 1), delta=(2, 4),
                  theta=(5, 7), alpha=(8, 12),
                  beta=(15, 29), gamma=(30, 50))
topos = dict()
stcs = dict()
topo_norm = evoked_psd.data.sum(axis=1, keepdims=True)
stc_norm = stc_psd.sum()
# Normalize each source point by the total power across freqs
for band, limits in freq_bands.items():
    data = evoked_psd.copy().crop(*limits).data.sum(axis=1, keepdims=True)
    topos[band] = mne.EvokedArray(100 * data / topo_norm, evoked_psd.info)
    stcs[band] = 100 * stc_psd.copy().crop(*limits).sum() / stc_norm.data

for freqs in ['ultra', 'delta', 'theta', 'alpha', 'beta', 'gamma']:
    fig, brain = plot_band(freqs)

# Get labels for HCPMMP1 cortical parcellation
labels = mne.read_labels_from_annot(
    'fsaverage', 'HCPMMP1', 'lh',
    subjects_dir=mne.datasets.sample.data_path() + '/subjects')

# Average the source estimates within each label of the cortical parcellation
# and each sub structures contained in the src space
# If mode = 'mean_flip' this option is used only for the cortical label
src = inverse_operator['src']
label_ts = mne.extract_label_time_course(stcs, labels, src,
                                         mode='mean_flip',
                                         allow_empty=True,
                                         return_generator=False)

# We compute the connectivity in the alpha band and plot it using a circular
# graph layout
fmin = freq_bands['alpha'][0]
fmax = freq_bands['alpha'][1]
sfreq = raw.info['sfreq']  # the sampling frequency
con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
    label_ts, method='pli', mode='multitaper', sfreq=sfreq, fmin=fmin,
    fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1)

# We create a list of Label containing also the sub structures
labels_aseg = mne.get_volume_labels_from_src(src, subject, subjects_dir)
labels = labels_parc + labels_aseg

# read colors
node_colors = [label.color for label in labels]

# We reorder the labels based on their location in the left hemi
label_names = [label.name for label in labels]
lh_labels = [name for name in label_names if name.endswith('lh')]
rh_labels = [name for name in label_names if name.endswith('rh')]

# Get the y-location of the label
label_ypos_lh = list()
for name in lh_labels:
    idx = label_names.index(name)
    ypos = np.mean(labels[idx].pos[:, 1])
    label_ypos_lh.append(ypos)
try:
    idx = label_names.index('Brain-Stem')
except ValueError:
    pass
else:
    ypos = np.mean(labels[idx].pos[:, 1])
    lh_labels.append('Brain-Stem')
    label_ypos_lh.append(ypos)


# Reorder the labels based on their location
lh_labels = [label for (yp, label) in sorted(zip(label_ypos_lh, lh_labels))]

# For the right hemi
rh_labels = [label[:-2] + 'rh' for label in lh_labels
             if label != 'Brain-Stem' and label[:-2] + 'rh' in rh_labels]

# Save the plot order
node_order = list()
node_order = lh_labels[::-1] + rh_labels

node_angles = circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) // 2])


# Plot the graph using node colors from the FreeSurfer parcellation. We only
# show the 300 strongest connections.
conmat = con[:, :, 0]
fig = plt.figure(num=None, figsize=(8, 8), facecolor='black')
plot_connectivity_circle(conmat, label_names, n_lines=300,
                         node_angles=node_angles, node_colors=node_colors,
                         title='All-to-All Connectivity left-Auditory '
                         'Condition (PLI)', fig=fig, interactive=False)
