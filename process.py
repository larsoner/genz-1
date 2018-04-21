# -*- coding: utf-8 -*-
"""
GenZ pilot analysis script.
@authors: larsoner & mdclarke

"""
#Notes:

#NOT COMPLETE
# genz301

#NO SSS
# genz504 & genz510 (exit 6)

#NO SCORE
# genz203: 66
# genz506: 48
# genz923: 120
# genz993: 120

#RuntimeError: Event time samples were not unique
# genz508

#NO MRI
# genz404

# skip 2,3,8,11,12,13,15

import mnefun
import numpy as np

params = mnefun.Params(tmin=None, tmax=None, n_jobs=18,
                       decim=1, proj_sfreq=200,
                       n_jobs_fir='cuda', n_jobs_resample='cuda',
                       filter_length='auto', lp_cut=80., bmin=None,
                       lp_trans='auto', bem_type='5120')
sub_ids = [103, 104, 201, 203, 302, 303, 305, 308, 404, 501, 502, 503,
           504, 506, 508, 510]
params.subjects = ['genz%d_ses1' %ss for ss in sub_ids]
params.subject_indices = np.arange(len(params.subjects))
params.dates = [None] * len(params.subjects)
bids_affix = '-1_fsmempr_ti1200_rms_1_freesurf_hires'
params.structurals = ['sub%s' % ss for ss in params.subjects]
params.subject_run_indices = None
params.subjects_dir = '/storage/Maggie/anat/subjects/'
params.score = None
params.run_names = ['%s_resting_01']
params.acq_ssh = 'kam@minea.ilabs.uw.edu'
params.acq_dir = ['/sinuhe/data01/genz', '/sinuhe/data03/genz']
params.sws_ssh = 'kambiz@kasga.ilabs.uw.edu'  # kasga
params.sws_dir = '/data07/kam/genz'
params.sss_type = 'python'
params.sss_regularize = 'in'
params.st_correlation = 0.98
params.trans_to = 'twa'
params.tsss_dur = 300.
params.movecomp = 'inter'
params.coil_t_window = 'auto'
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
# The ones we actually want
params.analyses = []
params.out_names = []
params.out_numbers = []
# do not trial count match for now
params.must_match = []
params.report_params.update(
    bem=True,
    psd=True,  # often slow
)
mnefun.do_processing(
    params,
    fetch_raw=True,
    do_score=False,
    push_raw=False,
    do_sss=True,
    fetch_sss=False,
    do_ch_fix=False,
    gen_ssp=True,
    apply_ssp=True,
    write_epochs=False,
    gen_covs=False,
    gen_fwd=False,
    gen_inv=False,
    gen_report=True,
    print_status=True,
)
