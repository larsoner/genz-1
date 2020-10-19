#!/usr/bin/env python

"""Compute narrow-band envelope correlation matrices
For each subject, epoch raw into 5 sec trials, compute erm covariance,
use Autoreject to clean up wide band epoched data. For each discrete frequency band
regularize covariance, compute inverse, compute pairwise power correlation between ROI 
label timeseries extracted from Freesurfer aparc_sub ROIs.
"""

import os
import os.path as op

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.sparse import csgraph

from meeg_preprocessing import config, utils
from mne import compute_raw_covariance
from mne.connectivity import envelope_correlation
from mne.filter import next_fast_len
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
fslabels = mne.read_labels_from_annot(
    "fsaverage", "aparc_sub", "both", subjects_dir=defaults.subjects_dir
)
fslabels = [label for label in fslabels if not label.name.startswith("unknown")]
label_nms = [rr.name for rr in fslabels]
n = len(label_nms)
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
        label_ts = funcs.extract_labels_timeseries(
            subject,
            raw,
            hp,
            lp,
            cov,
            fwd,
            defaults.subjects_dir,
            eps_fname,
            fslabels,
            return_generator=True,
        )
        aec = envelope_correlation(label_ts)
        assert aec.shape == (len(fslabels), len(fslabels))
        _, deg = csgraph.laplacian(aec, return_diag=True)
        ###############ref Cedric & Stack#############
        data[si, ix] = deg
foo = funcs.expand_grid({"id": df.id.values, "freq": defaults.bands, "roi": label_nms})
foo["deg"] = pd.Series(data.flatten())
foo.to_csv(op.join(defaults.payload, "nxLaplnsXroi-tidy.csv"))
# bar = foo.pivot_table("deg", "id", ["freq", "roi"], aggfunc="first").to_csv(
#     op.join(defaults.payload, "nxLaplnsXroi-wide.csv")
# )
