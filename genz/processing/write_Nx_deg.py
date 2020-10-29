#!/usr/bin/env python

"""Compute narrow-band envelope correlation matrices
For each subject, epoch raw MEG into 5 sec. long arbitrary trials, 
use Autoreject to clean up wide band epoched data. Compute erm covariances.
For each frequency band regularize data covariance, compute inverse, 
compute pairwise power envelope correlation between FS aparc_sub ROI labels.
"""

import os
import os.path as op
import warnings
            
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.sparse import csgraph

from meeg_preprocessing import config, utils
from mne import compute_raw_covariance
from mne.connectivity import envelope_correlation
from mne.filter import next_fast_len
from autoreject import AutoReject
from meeg_preprocessing import config
from mne import read_epochs
from mne.cov import regularize
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mnefun import get_fsaverage_medial_vertices

from genz import defaults, funcs

### locals
new_sfreq = defaults.new_sfreq
n_fft = next_fast_len(int(round(4 * new_sfreq)))
lims = [75, 85, 95]
medial_verts = get_fsaverage_medial_vertices()
dfs = []
for ag in [9, 11, 13, 15, 17]:
    fi = op.join(defaults.static, "GenZ_subject_information - %da group.tsv" % ag)
    dfs.append(pd.read_csv(fi, sep="\t", usecols=["Subject Number", "Sex"]))
df = pd.concat(dfs)
df.columns = ["id", "sex"]
df.sort_values(by="id")
dups = df[df.duplicated("id")].index.values.tolist()
df.drop(df.index[dups], inplace=True)
df.drop(df[df.id.isin(defaults.exclude)].index, inplace=True)
df = df.dropna(how="all")
rois = mne.read_labels_from_annot(
    "fsaverage", "aparc_sub", "both", subjects_dir=defaults.subjects_dir
)
rois = [roi for roi in rois if not roi.name.startswith("unknown")]
roi_nms = [rr.name for rr in rois]
n = len(roi_nms)
data = np.zeros((len(df), len(defaults.bands), n))
###start subject loop
for si, ss in enumerate(df.id.values):
    subject = ss.replace("GenZ_", "genz")
    print("    Loading data for %s" % subject)
    bem_dir = os.path.join(defaults.subjects_dir, subject, "bem")
    bem_fname = os.path.join(bem_dir, "%s-5120-bem-sol.fif" % subject)
    src_fname = os.path.join(bem_dir, "%s-oct-6-src.fif" % subject)
    subj_dir = os.path.join(defaults.megdata, subject)
    raw_fname = os.path.join(
        subj_dir, "sss_pca_fif", "%s_rest_01_allclean_fil100_raw_sss.fif" % subject
    )
    erm_fname = os.path.join(
        subj_dir, "sss_pca_fif", "%s_erm_01_allclean_fil100_raw_sss.fif" % subject
    )
    trans_fname = os.path.join(subj_dir, "trans", "%s-trans.fif" % subject)
    eps_dir = os.path.join(subj_dir, "epochs")
    eps_fname = op.join(eps_dir, "All_sss_%s-epo.fif" % subject)
    src_dir = os.path.join(subj_dir, "source")
    # Load raw
    try:
        raw = mne.io.read_raw_fif(raw_fname)
    except FileNotFoundError as e:
        continue
    if not raw.info["highpass"] == 0:
        print("%s acquisition HP greater than DC" % subject)
        continue
    if not os.path.exists(eps_dir):
        os.mkdir(eps_dir)
    if not os.path.exists(src_dir):
        os.mkdir(src_dir)
    raw.load_data().resample(new_sfreq, n_jobs=config.N_JOBS)
    raw_erm = mne.io.read_raw_fif(erm_fname)
    raw_erm.load_data().resample(new_sfreq, n_jobs=config.N_JOBS)
    raw_erm.add_proj(raw.info["projs"])
    # ERM covariance
    cov = compute_raw_covariance(
        raw_erm,
        n_jobs=config.N_JOBS,
        reject=dict(grad=4000e-13, mag=4e-12),
        flat=dict(grad=1e-13, mag=1e-15),
    )
    # Make forward stack and get transformation matrix
    src = mne.read_source_spaces(src_fname)
    bem = mne.read_bem_solution(bem_fname)
    trans = mne.read_trans(trans_fname)
    fwd = mne.make_forward_solution(
        raw.info, trans, src=src, bem=bem, eeg=False, verbose=True
    )
    # epoch raw into 5 sec trials
    print("      \nLoading ...%s" % op.relpath(eps_fname, defaults.megdata))
    ###start bands loop
    for ix, (kk, vv) in enumerate(defaults.bands.items()):
        hp, lp = vv
        # epoch raw into 5 sec trials
        events = mne.make_fixed_length_events(raw, duration=5.)
        epochs = mne.Epochs(raw, events=events, tmin=0, tmax=5.,
                            baseline=None, reject=None, preload=True)
        if not op.isfile(eps_fname):
            # k-fold CV thresholded artifact rejection
            ar = AutoReject()
            epochs = ar.fit_transform(epochs)
            print('      \nSaving ...%s' % op.relpath(eps_fname,
                                                    defaults.megdata))
            epochs.save(eps_fname, overwrite=True)
        epochs = read_epochs(eps_fname)
        print('%d, %d (Epochs, drops)' %
            (len(events), len(events) - len(epochs.selection)))
        #epochs.plot_psd()
        roi_nms = np.setdiff1d(np.arange(len(events)), epochs.selection)
        # raw = raw.copy().filter(lf, hf, fir_window='blackman',
        #                       method='iir', n_jobs=config.N_JOBS)
        iir_params = dict(order=4, ftype='butter', output='sos')
        epochs_ = epochs.copy().filter(hp, lp, method='iir',
                                    iir_params=iir_params,
                                    n_jobs=config.N_JOBS)
        # epochs_.plot_psd(average=True, spatial_colors=False)
        mne.Info.normalize_proj(epochs_.info)
        # epochs_.plot_projs_topomap()
        # regularize covariance
        # rank = compute_rank(cov, rank='full', info=epochs_.info)
        cov = regularize(cov, raw.info)
        inv = make_inverse_operator(epochs_.info, fwd, cov)
        # Compute label time series and do envelope correlation
        stcs = apply_inverse_epochs(epochs_, inv, lambda2=1. / 9.,
                                    pick_ori='normal',
                                    return_generator=True)
        morphed = mne.morph_labels(rois, subject,
                                subjects_dir=defaults.subjects_dir)
        label_ts = mne.extract_label_time_course(stcs, morphed, fwd['src'],
                                            return_generator=True,
                                            verbose=True)
        aec = envelope_correlation(label_ts)
        assert aec.shape == (len(rois), len(rois))
        _, deg = csgraph.laplacian(aec, return_diag=True)
        ###############ref Cedric & Stack#############
        data[si, ix] = deg
        threshold_prop = 0.15  # percentage of strongest edges to keep in the graph
<<<<<<< HEAD
        degree = mne.connectivity.degree(aec, threshold_prop=threshold_prop)
        if not np.allclose(deg, degree):
            warnings.warn('mne.connectivity.degree NOT equal to csgraph.laplacian')
        stc = mne.labels_to_stc(rois, degree)
=======
        degree = mne.connectivity.degree(corr, threshold_prop=threshold_prop)
        if not np.allclose(deg, degree):
            warnings.warn('mne.connectivity.degree NOT equal to csgraph.laplacian')
        stc = mne.labels_to_stc(labels, degree)
>>>>>>> cf4b12c14e3f5556241a9a743f259724aac9308f
        stc = stc.in_label(mne.Label(inv['src'][0]['vertno'], hemi='lh') +
                        mne.Label(inv['src'][1]['vertno'], hemi='rh'))
        brain = stc.plot(
            clim=dict(kind='percent', lims=[75, 85, 95]), colormap='gnuplot',
<<<<<<< HEAD
            subjects_dir=defaults.subjects_dir, views='dorsal', hemi='both',
            smoothing_steps=25, time_label='%s band' % kk)
        brain.savefig(op.join(defaults.payload, 'degree-%s.png' % kk))
foo = funcs.expand_grid({"id": df.id.values, "freq": defaults.bands, "roi": roi_nms})
=======
            subjects_dir=subjects_dir, views='dorsal', hemi='both',
            smoothing_steps=25, time_label='%s band' % kk)
        brain.savefig(op.join(defaults.payload, 'degree-%s.png' % kk))
foo = funcs.expand_grid({"id": df.id.values, "freq": defaults.bands, "roi": label_nms})
>>>>>>> cf4b12c14e3f5556241a9a743f259724aac9308f
foo["deg"] = pd.Series(data.flatten())
foo.to_csv(op.join(defaults.payload, "degree_x_frequency-roi.csv"))
# bar = foo.pivot_table("deg", "id", ["freq", "roi"], aggfunc="first").to_csv(
#     op.join(defaults.payload, "nxLaplnsXroi-wide.csv")
# )
