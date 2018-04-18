# -*- coding: utf-8 -*-

import sys
import os.path as op
import mne
from mne import (pick_types, compute_raw_covariance)
from mne.preprocessing import (maxwell_filter, ICA, compute_proj_ecg,
                               compute_proj_eog)
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
from mne import setup_source_space
from mne import make_forward_solution

print('\n'.join(sys.path))
subjects_dir = '/Users/ktavabi/Data/freesufer'
n_jobs = 2
raw_fname = '/Users/ktavabi/Data/genzds/genz_bo_resting_01_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True,
                          allow_maxshield='yes')
head_pos = mne.chpi._calculate_chpi_positions(raw=raw)
mne.viz.plot_head_positions(head_pos, mode='traces',
                            destination=raw.info['dev_head_t'],
                            info=raw.info,
                            show=True)
# Denoise
if not op.isfile(raw_fname[:-4] + '_sss.fif'):
    raw = mne.chpi.filter_chpi(raw)
    raw.fix_mag_coil_types()
    raw_sss = maxwell_filter(raw, head_pos=head_pos, st_duration=300)
    raw_sss.save(raw_fname[:-4] + '_sss.fif')
else:
    raw_sss = mne.io.read_raw_fif(raw_fname[:-4] + '_sss.fif', preload=True)

# artifact correction
ecg_projs, events = compute_proj_ecg(raw_sss, n_grad=3, n_mag=3, n_eeg=0,
                                     average=True)
print(ecg_projs)
mne.viz.plot_projs_topomap(ecg_projs)
eog_projs, events = compute_proj_eog(raw_sss, n_grad=3, n_mag=3, n_eeg=0,
                                     average=True)
print(eog_projs)
mne.viz.plot_projs_topomap(eog_projs)
raw_sss.info['projs'] += eog_projs + ecg_projs
picks = pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False,
                   exclude='bads')
reject = dict(mag=5e-12, grad=4000e-13)
raw_filt = raw_sss.copy().pick_types(meg=True, eeg=False,
                                     stim=False, eog=False,
                                     exclude='bads')
raw_filt.filter(1, 4, fir_design='firwin', n_jobs=n_jobs)
raw_cov = compute_raw_covariance(raw_filt)

# Compute the fwd matrix
bem_dir = op.join(subjects_dir, 'genz_bo', 'bem')
bem_fname = op.join(bem_dir, 'genz_bo-5120-bem-sol.fif')
src = mne.setup_source_space('genz_bo', subjects_dir=subjects_dir)
fwd = make_forward_solution(raw_fname, 'genz_bo-trans.fif', src, bem_fname,
                            mindist=5.0,  # ignore sources<=5mm from innerskull
                            meg=True, eeg=False,
                            n_jobs=6)
# Setup a surface-based source space
subjects_dir = mne.get_config('SUBJECTS_DIR')
src = setup_source_space('genz_bo', subjects_dir=subjects_dir,
                         spacing='oct6', add_dist=False)

# Look at alignments
trans = mne.read_trans('genz_bo-trans.fif')
mne.viz.plot_alignment(raw.info, trans=trans, subject='genz_bo', src=src,
                       subjects_dir=subjects_dir, surfaces=['head', 'white'],
                       show=True)
lcmv_ws = mne.beamformer.make_lcmv


# ICA
ica = ICA(n_components=0.9,  method='fastica', noise_cov=raw_cov)
# projs, raw_filt.info['projs'] = raw_sss.info['projs'], []
S_ = ica.fit(raw_filt, reject=reject)
# raw_filt.info['projs'] = projs
h = S_.plot_components(res=128, cmap='viridis', inst=raw_filt)
S_.plot_properties(raw_filt, picks=range(5))

raw_copy = raw_filt.copy()
S_.apply(raw_copy)
raw_copy.plot()  # check the result

# Compute inverse operator
snr = 1.0  # use smaller SNR for raw data
lambda2 = 1.0 / snr ** 2
method = "sLORETA"  # use sLORETA method (could also be MNE or dSPM)
inverse_op = make_inverse_operator(raw_copy.info, fwd, raw_cov)
# Compute inverse solution
start, stop = raw_copy.time_as_index([0, 30])
stc = apply_inverse_raw(raw_copy, inverse_op, lambda2, method,
                        start=start, stop=stop, pick_ori=None)

brain = stc.plot('genz_bo', 'inflated', 'split',
                 subjects_dir=subjects_dir, time_viewer=True)
brain.show_view('lateral')
