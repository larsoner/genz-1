# -*- coding: utf-8 -*-

"""Compute LCMV inverse solution on rsMEG dataset in a volume source space.
Store the solution in a nifti file for visualisation, e.g. with Freeview """

# Author(s): Kambiz Tavabi <ktavabi@gmail.com>

import time
import copy
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.io import read_raw_fif
from mne.chpi import _calculate_chpi_positions
from mne.preprocessing import (maxwell_filter, compute_proj_ecg,
                               compute_proj_eog)
from mne.viz import plot_projs_topomap
from mne.beamformer import make_lcmv, apply_lcmv_raw
from mne.utils import get_subjects_dir, run_subprocess
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img


def process_raw(raw_fname):
    raw = read_raw_fif(raw_fname, preload=True, allow_maxshield='yes')
    head_pos = _calculate_chpi_positions(raw=raw)
    raw = mne.chpi.filter_chpi(raw)
    raw.fix_mag_coil_types()
    raw_sss = maxwell_filter(raw, head_pos=head_pos, st_duration=300)
    raw_sss.save(raw_fname[:-4] + '_sss.fif')
    return raw, head_pos


def do_pca(raw, proj_nums):
    ecg_projs, ecg_events = compute_proj_ecg(raw_sss, n_grad=proj_nums[0][0],
                            n_mag=proj_nums[0][1],
                            average=True)

    eog_projs, eog_events = compute_proj_eog(raw_sss, n_grad=proj_nums[1][0],
                            n_mag=proj_nums[1][1],
                            average=True)
    raw_sss.info['projs'] += eog_projs + ecg_projs
    return raw_sss, (ecg_projs, eog_projs, ecg_events, eog_events)


environ = copy.copy(os.environ)
subjects_dir = mne.utils.get_subjects_dir(raise_error=True)
datapath = '/media/ktavabi/INDAR/data/genz/rsMEG/'
subject = 'genz501_17a'
raw_fname = op.join(datapath, subject, 'raw_fif',
                    '%s_rest_01_raw.fif' % subject)
sss_fname = op.join(datapath, subject, 'sss_fif',
                    '%s_rest_01_raw_sss.fif' % subject)
n_jobs = 12
proj_nums = [[1, 2],  # ECG [grad, mag]
             [2, 2]]  # EOG

# Denoise & clean
if not op.isfile(sss_fname):
    raw_sss, head_pos = process_raw(raw_fname)
else:
    raw_sss = read_raw_fif(sss_fname, preload=True)

raw_sss_pca, pca = do_pca(raw_sss, proj_nums)
ecg_projs, eog_projs, ecg_events, eog_events = pca
print(ecg_projs[-2:])
print(eog_projs[-2:])
plot_projs_topomap(ecg_projs[-2:])
plot_projs_topomap(eog_projs[-2:])

# Filter
raw_filt = raw_sss_pca.copy().filter(h_freq=4., l_freq=None,
                                     filter_length='30s', n_jobs=n_jobs,
                                     fir_design='firwin2', phase='zero-double')

# Pick the channels of interest
raw_filt.pick_types(meg='grad')

# Re-normalize projectors after subselection
raw_filt.info.normalize_proj()

# regularized data covariance
data_cov = mne.compute_raw_covariance(raw_filt, n_jobs=n_jobs)

# beamformer requirements
bem = op.join(subjects_dir, subject, 'bem', 'genz501_17a-5120-bem-sol.fif')
sphere = mne.make_sphere_model(r0='auto', head_radius='auto',
                               info=raw_filt.info)
src = mne.setup_volume_source_space(subject='fsaverage', bem=bem,
                                    mri=op.join(subjects_dir, 'fsaverage',
                                                'mri', 'T1.mgz')
                                    subjects_dir=subjects_dir)
fwd = mne.make_forward_solution(raw_filt.info, trans=None, src=src,
                                bem=bem, n_jobs=n_jobs)
filters = make_lcmv(raw_filt.info, fwd, data_cov, reg=0.05,
                    pick_ori='max-power', weight_norm='nai',
                    reduce_rank=True)
t0 = time.time()
stc = apply_lcmv_raw(raw_filt, filters)
print(' Time: %s mns' % round((time.time() - t0) / 60, 2))

# Save result in stc files
stc.save(op.join(datapath, subject, 'lcmv-vol'))
stc.crop(0.0, 1.0)
# plot dSPM time course in src space
kwargs = dict(color='c', linestyle='--', linewidth=.5)
f, ax = plt.subplots(1, 1, figsize=(8, 11))
mx = np.argmax(stc.data, axis=0)
ax.plot(stc.times, stc.data[mx[:100], :].T)
ax.autoscale(enable=True, axis='x', tight=True)
ax.grid(True, which='major', axis='y', **kwargs)
ax.set_xlabel('time (s)')
ax.set_ylabel('strength')

# Save result in a 4D nifti file
img = mne.save_stc_as_volume(op.join(datapath, subject, 'lcmv_inverse.nii.gz',),
                             stc, fwd['src'], mri_resolution=False)

t1_fname = subjects_dir + '/fsaverage/mri/T1.mgz'

# Plotting with nilearn
idx = stc.time_as_index(0.5)
plot_stat_map(index_img(img, idx), t1_fname, threshold=0.45,
              title='LCMV (t=%.3f s.)' % stc.times[idx])

# plot source time courses with the maximum peak amplitudes at idx
plt.figure()
plt.plot(stc.times, stc.data[np.argsort(np.max(stc.data[:, idx],
                                               axis=1))[-40:]].T)
plt.xlabel('Time (ms)')
plt.ylabel('LCMV value')
plt.show()
