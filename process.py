# -*- coding: utf-8 -*-
"""
GenZ pilot analysis script.
@author: Kambiz Tavabi
@contact: ktavabi@gmail.com
@license: MIT
@date: 04/21/2018
"""

import os.path as op
import mnefun
import numpy as np
from picks import names, bad_channels

params = mnefun.Params(tmin=None, tmax=None, n_jobs=18,
                       decim=1, proj_sfreq=200,
                       n_jobs_fir='cuda', n_jobs_resample='cuda',
                       filter_length='auto', lp_cut=80., bmin=None,
                       lp_trans='auto', bem_type='5120')


params.subjects = names
params.subject_indices = np.setdiff1d(np.arange(len(params.subjects)),
                                      np.array([3]))
# write prebads
for si, subj in enumerate(params.subjects):
    if op.exists(op.join(params.work_dir, subj, 'raw_fif')):
        prebad_file = op.join(params.work_dir, subj,
                              'raw_fif', '%s_prebad.txt' % subj)
        if not op.exists(prebad_file):
            assert len(bad_channels) == len(params.subjects)
            if bad_channels[si] is None:
                with open(prebad_file, 'w') as f:
                    f.write("")
            else:
                with open(prebad_file, 'w') as output:
                    for ch_name in bad_channels[si]:
                        output.write("%s\n" % ch_name)

params.dates = [None] * len(params.subjects)
params.structurals = ['sub%s' % ss for ss in params.subjects]
params.subject_run_indices = None
params.subjects_dir = '/brainstudio/data/genz/freesurf_subjs'
params.score = None
params.run_names = ['%s_rest_01']
params.acq_ssh = 'kambiz@minea.ilabs.uw.edu'
params.acq_dir = ['/sinuhe/data01/genz', '/sinuhe/data03/genz']
params.sws_ssh = 'kam@kasga.ilabs.uw.edu'  # kasga
params.sws_dir = '/data07/kam/genz'
params.sss_type = 'python'
params.sss_regularize = 'in'
params.st_correlation = 0.98
params.trans_to = 'twa'
params.tsss_dur = 300.
# Set the parameters for head position estimation:
params.coil_dist_limit = 0.01
params.coil_t_window = 'auto'  # use the smallest reasonable window size
# remove segments with < 3 good coils for at least 1 sec
params.coil_bad_count_duration_limit = 1.  # sec
# Annotation params
params.rotation_limit = 20.  # deg/s
params.translation_limit = 0.01  # m/s
# Trial rejection
params.reject = dict()
params.proj_ave = True
params.flat = dict(grad=1e-13, mag=1e-15)
# Which runs and trials to use
params.get_projs_from = np.arange(1)
params.inv_names = ['%s']
params.inv_runs = [np.arange(1)]
params.proj_nums = [[3, 3, 0],  # ECG: grad/mag/eeg
                    [3, 3, 0],  # EOG
                    [0, 0, 0]]  # Continuous (from ERM)
params.on_missing = 'ignore'  # some subjects will not complete the paradigm
params.report_params.update(
    bem=True,
    psd=True,  # often slow
)
mnefun.do_processing(
    params,
    fetch_raw=False,
    do_score=False,
    push_raw=False,
    do_sss=True,
    fetch_sss=False,
    do_ch_fix=False,
    gen_ssp=False,
    apply_ssp=False,
    write_epochs=False,
    gen_covs=False,
    gen_fwd=False,
    gen_inv=False,
    gen_report=False,
    print_status=True,
)
