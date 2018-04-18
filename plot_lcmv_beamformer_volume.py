# -*- coding: utf-8 -*-

"""Compute LCMV inverse solution on rsMEG dataset in a volume source space.
Store the solution in a nifti file for visualisation, e.g. with Freeview """

# Author(s): Kambiz Tavabi <ktavabi@gmail.com>

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


def remove_artifacts_pca(raw, proj_nums):
    ecg_projs, _ = compute_proj_ecg(raw_sss, n_grad=proj_nums[0][0],
                                    n_mag=proj_nums[0][1],
                                    average=True)

    eog_projs, _ = compute_proj_eog(raw_sss, n_grad=proj_nums[1][0],
                                    n_mag=proj_nums[1][1],
                                    average=True)
    raw_sss.info['projs'] += eog_projs + ecg_projs
    return raw_sss, ecg_projs, eog_projs


subjects_dir = '/Users/ktavabi/Data/freesufer'
datapath = '/Users/ktavabi/Data/genzds/'
raw_fname = datapath + 'genz_lily_resting_01_raw.fif'
n_jobs = 2
proj_nums = [[1, 1, 0],  # ECG
             [1, 1, 0]]  # EOG

# Denoise & clean
if not op.isfile(raw_fname[:-4] + '_sss.fif'):
    raw_sss, head_pos = process_raw(raw_fname)
else:
    raw_sss = read_raw_fif(raw_fname[:-4] + '_sss.fif', preload=True)

raw_sss_pca, ecg_projs, eog_projs = remove_artifacts_pca(raw_sss, proj_nums)
print(ecg_projs[-2:])
plot_projs_topomap(ecg_projs[-2:])
print(eog_projs[-2:])
plot_projs_topomap(eog_projs[-2:])

# Pick the channels of interest
raw_sss_pca.pick_types(meg='grad')

# Re-normalize projectors after subselection
raw_sss_pca.info.normalize_proj()

# regularized data covariance
data_cov = mne.compute_raw_covariance(raw_sss_pca, n_jobs=n_jobs)

# beamformer requirements
sphere = mne.make_sphere_model(r0='auto', head_radius='auto',
                               info=raw_sss_pca.info)
src = mne.setup_volume_source_space(subject='genz_lily', sphere=sphere,
                                    subjects_dir=subjects_dir)
fwd = mne.make_forward_solution(raw_sss_pca.info, trans=None, src=src,
                                bem=sphere, n_jobs=n_jobs)
filters = make_lcmv(raw_sss_pca.info, fwd, data_cov, reg=0.05,
                    pick_ori='max-power', weight_norm='nai',
                    reduce_rank=True)
stc = apply_lcmv_raw(raw_sss_pca.copy().filter(), filters)
stc.data[:, :] = np.abs(stc.data)

# Save result in stc files
stc.save('lcmv-vol')
stc.crop(0.0, 2.5)

# Save result in a 4D nifti file
img = mne.save_stc_as_volume(datapath + 'genz_lily_lcmv_inverse.nii.gz',
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
