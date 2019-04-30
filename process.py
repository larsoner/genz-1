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
import pandas as pd
import numpy as np

params = mnefun.Params(n_jobs=18,
                       decim=1, proj_sfreq=500,
                       n_jobs_fir='cuda', n_jobs_resample='cuda',
                       filter_length='auto', lp_cut=100.,
                       lp_trans='auto', bem_type='5120',
                       tmin=-.2, tmax=.2)

# write prebads
picks = pd.read_csv('/home/ktavabi/Github/genz/static/picks.tsv', sep='\t',
                    usecols=['id', 'badChs'])
exclude = ['104_9a',  # Too few EOG events
           '108_9a',  # Fix
           '113_9a',  # Too few ECG events
           '115_9a',  # no cHPI
           '209_11a',  # Too few EOG events
           '231_11a',  # twa_hp calc fail with assertion error
           '432_15a',  # Too few ECG events
           '510_17a',  # Too few ECG events
           '527_17a']  # Too few EOG events
picks.drop(picks[picks.id.isin(exclude)].index, inplace=True)
picks.sort_values(by='id', inplace=True)
for si, subj in enumerate(picks.id.values):
    subj = 'genz%s' % subj
    if op.exists(op.join(params.work_dir, subj, 'raw_fif')):
        prebad_file = op.join(params.work_dir, subj,
                              'raw_fif', '%s_prebad.txt' % subj)
        if not op.exists(prebad_file):
            print('Writing channles %s to prebad for %s'
                  % (picks.iloc[si].badChs, subj))
            if picks.iloc[si].badChs is None:
                with open(prebad_file, 'w') as output:
                    output.write("")
            elif len(picks.iloc[si].badChs.split(sep=',')) == 1:
                with open(prebad_file, 'w') as output:
                   output.write("MEG%s\n" % picks.iloc[si].badChs)
            else:
                with open(prebad_file, 'w') as output:
                    for ch_name in picks.iloc[si].badChs.split(sep=','):
                        output.write("MEG%s\n" % ''.join(ch_name.split()))

params.subjects = ['genz%s' % cc for cc in picks.id.values]
params.subject_indices = np.setdiff1d(np.arange(len(params.subjects)),
                                      np.array([3]))
params.dates = [None] * len(params.subjects)
params.structurals = params.subjects
params.subject_run_indices = None
params.subjects_dir = '/mnt/jaba/meg/genz/anatomy'
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
params.tsss_dur = 299.
# Set the parameters for head position estimation:
params.coil_dist_limit = 0.01
params.coil_t_window = 'auto'  # use the smallest reasonable window size
# remove segments with < 3 good coils for at least 1 sec
params.coil_bad_count_duration_limit = 1.  # sec
# Annotation params
params.rotation_limit = 20.  # deg/s
params.translation_limit = 0.01  # m/s
# Cov
params.runs_empty = ['%s_erm_01']  # Define empty room runs
params.cov_method = 'shrunk'
params.compute_rank = True  # compute rank of the noise covariance matrix
params.force_erm_cov_rank_full = False  # compute and use the empty-room rank
params.reject = dict()
params.flat = dict(grad=1e-13, mag=1e-15)
# Proj
params.get_projs_from = np.arange(1)
params.proj_ave = True
params.proj_meg = 'combined'
params.inv_names = ['%s']
params.inv_runs = [np.arange(1)]
params.proj_nums = [[1, 1, 0],  # ECG: grad/mag/eeg
                    [1, 1, 0],  # EOG
                    [1, 1, 0]]  # Continuous (from ERM)
params.report_params.update(
        bem=True,
        psd=True,  # often slow
        ssp=True,
        source_alignment=True
        
        )
mnefun.do_processing(
        params,
        fetch_raw=False,
        do_score=False,
        push_raw=False,
        do_sss=False,
        fetch_sss=False,
        do_ch_fix=False,
        gen_ssp=False,
        apply_ssp=False,
        write_epochs=False,
        gen_covs=False,
        gen_fwd=False,
        gen_inv=False,
        gen_report=True,
        print_status=False,
        )
