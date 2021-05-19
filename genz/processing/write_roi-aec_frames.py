#!/usr/bin/env python

"""Compute narrow-band functional connectomes
For each subject, epoch raw .fif resting state into 5 sec. long arbitrary
trials data cleaned-up using Autoreject. Do source imaging using ERM SSP,
for each bandwidth (n=7) compute regularized ROI AEC covariance array.
"""

import os
import os.path as op
import time

import mne
import numpy as np
import pandas as pd
from genz import defaults, funcs
from mne.connectivity import envelope_correlation
from mne.externals import h5io
from mne.minimum_norm import apply_inverse_epochs
from scipy.sparse import csgraph
import mnefun
from mnefun import get_raw_fnames
from mnefun._epoching import _concat_resamp_raws

dfs = []
for ag in defaults.ages:
    fi = op.join(
        defaults.static, "GenZ_subject_information - %da group.tsv" % ag
    )
    dfs.append(pd.read_csv(fi, sep="\t", usecols=["Subject Number", "Sex"]))
df = pd.concat(dfs)
df.reset_index(drop=True, inplace=True)
df.columns = ["id", "sex"]
df.drop_duplicates("id", inplace=True)
df = df.dropna(how="any")
df.sort_values("id")
df.id[:] = [
    id_.split("_", 1)[1] if isinstance(id_, str) else id_ for id_ in df.id
]
exclude = df.id.isin(defaults.exclude)
df = df[~exclude]
include = df.id.isin(defaults.include)
df = df[include]
assert len(df) == 106, len(df)

all_rois = mne.read_labels_from_annot(
    "fsaverage", "aparc_sub", "both", subjects_dir=defaults.subjects_dir
)
rois = [roi for roi in all_rois if not roi.name.startswith("unknown")]
roi_nms = [rr.name for rr in rois]
n = len(roi_nms)
deg_lap = np.zeros((len(df), len(defaults.bands), n))
degree = np.zeros_like(deg_lap)
A_lst = list()
reject = dict(grad=2000e-13, mag=6000e-15)  # same as GenZ repo
src_fs = mne.read_source_spaces(
    op.join(
        defaults.subjects_dir, "fsaverage", "bem", "fsaverage-ico-5-src.fif"
    )
)
state = "task"  # task/rest
if state == "task":
    p = mnefun.Params()
    p.work_dir = defaults.megdata
    p.sss_fif_tag = "_raw_sss.fif"
    p.run_names = [
        "%s_faces_learn_01",
        "%s_thumbs_learn_01",
        "%s_emojis_learn_01",
        "%s_faces_test_01",
        "%s_thumbs_test_01",
        "%s_emojis_test_01",
    ]
    p.lp_cut = 80

# Subject loop
for si, ss in enumerate(df.id.values):
    subject = f"genz{ss}"
    del ss
    subj_dir = os.path.join(defaults.megdata, subject)
    inv_fname = os.path.join(subj_dir, "inverse", f"{subject}-meg-erm-inv.fif")
    # Load raw
    print("Loading data for %s" % subject)
    if state == "task":
        p.pca_dir = os.path.join(subj_dir, "sss_pca_fif")
        raws = get_raw_fnames(p, subject, which="pca")
        if len(raws) == 0:
            continue
        raw = _concat_resamp_raws(p, subject, raws)[0]
    else:
        raw_fname = os.path.join(
            subj_dir,
            "sss_pca_fif",
            f"{subject}_rest_01_allclean_fil80_raw_sss.fif",
        )
        try:
            raw = mne.io.read_raw_fif(raw_fname)
        except FileNotFoundError as e:
            print(f"    File not found: {raw_fname}")
        continue
    # 0.1 should be okay for 5 sec epochs (envcorr will baseline correct
    # essentially because it's a correlation)
    if raw.info["highpass"] > 0.11:
        print(
            f"{subject} acquisition HP greater than 0.1 Hz "
            f"({raw.info['highpass']})"
        )
        continue
    raw.load_data()

    src_dir = os.path.join(subj_dir, "source")
    if not os.path.exists(src_dir):
        os.mkdir(src_dir)
    out_fname = os.path.join(src_dir, f"{state}_envcorr.h5")
    if op.isfile(out_fname):
        data = h5io.read_hdf5(out_fname)
        degree[si] = data["degree"]
        deg_lap[si] = data["deg_lap"]
        continue

    # epoch raw into 5 sec trials
    print("    Loading epochs ...", end="")
    a_lst = dict()
    events = mne.make_fixed_length_events(raw, duration=5.0)
    tmax = 5.0 - 1.0 / defaults.new_sfreq
    decim = 4
    # first create originals to get drops via peak-to-peak rejection
    epochs = mne.Epochs(
        raw,
        events=events,
        tmin=0,
        tmax=tmax,
        baseline=None,
        reject=reject,
        preload=True,
        decim=4,
    )
    assert epochs.info["sfreq"] == defaults.new_sfreq
    print(f" Dropped {len(events) - len(epochs)}/{len(events)} epochs")
    events = epochs.events
    # Network loop
    for ix, (kk, vv) in enumerate(defaults.bands.items()):
        print(f'    Processing {"-".join(str(v) for v in vv)} Hz ...', end="")
        t0 = time.time()
        hp, lp = vv
        a_lst[kk] = list()
        # now filter raw and re-epoch
        hp = None if hp == 0 else hp
        print(" Filtering ...", end="")
        raw_use = raw.copy().filter(
            hp,
            lp,
            l_trans_bandwidth=1,
            h_trans_bandwidth=1,
            n_jobs=defaults.n_jobs,
        )
        epochs = mne.Epochs(
            raw_use,
            events=epochs.events,
            tmin=0,
            tmax=tmax,
            baseline=None,
            reject=None,
            preload=True,
            decim=decim,
        )
        epochs.apply_hilbert(envelope=False)

        # Compute ROI time series and do envelope correlation
        inv = mne.minimum_norm.read_inverse_operator(inv_fname)
        these_rois = mne.morph_labels(
            all_rois, subject, "fsaverage", defaults.subjects_dir
        )
        these_rois = [
            roi for roi in these_rois if not roi.name.startswith("unknown")
        ]
        stcs = apply_inverse_epochs(
            epochs,
            inv,
            lambda2=1.0 / 9.0,
            pick_ori="normal",
            return_generator=True,
        )
        label_ts = mne.extract_label_time_course(
            stcs, these_rois, inv["src"], return_generator=True
        )

        # compute ROI level envelope power
        print(" Envcorr ...", end="")
        aec = envelope_correlation(label_ts)
        assert aec.shape == (len(rois), len(rois))
        # compute ROI laplacian as per:
        # Ginestet, C. E., Li, J., Balachandran, P., Rosenberg, S., &
        # Kolaczyk, E. D. (2017). Hypothesis testing for network data
        # in functional neuroimaging. Annals of Applied Statistics,
        # 11(2), 725–750. https://doi.org/10.1214/16-AOAS1015
        _, deg_lap[si, ix] = csgraph.laplacian(aec, return_diag=True)

        # compute ROI degree
        degree[si, ix] = mne.connectivity.degree(aec, threshold_prop=0.2)
        # if not np.allclose(deg_lap, degree[si, ix]):
        #     warnings.warn("mne.connectivity.degree NOT equal to laplacian")
        print(f" Completed in {(time.time() - t0) / 60:0.1f} min")
        h5io.write_hdf5(
            out_fname,
            dict(degree=degree[si], deg_lap=deg_lap[si]),
            overwrite=True,
        )

# visualize connectivity on fsaverage
for ix, (kk, vv) in enumerate(defaults.bands.items()):
    grab = np.mean(degree[:, ix], axis=0)  # avg across subjects for this band
    this_stc = mne.labels_to_stc(rois, grab, src=src_fs)
    assert this_stc.data.shape == (20484, 1)
    brain = this_stc.plot(
        subject="fsaverage",
        clim=dict(kind="percent", lims=[75, 85, 95]),
        colormap="gnuplot",
        subjects_dir=defaults.subjects_dir,
        views="dorsal",
        hemi="both",
        time_viewer=False,
        show_traces=False,
        smoothing_steps="nearest",
        time_label="%s band" % kk,
    )
    brain.save_image(
        op.join(defaults.payload, "%s-%s-nxDegree-group-roi.png" %(state, kk)
    )

# write out network laplacian data
laplacians = funcs.expand_grid(
    {"id": df.id.values, "freq": defaults.bands, "roi": roi_nms}
)
laplacians["nabla"] = pd.Series(deg_lap.flatten())
laplacians.to_csv(
    op.join(defaults.payload, f"{state}-Nx-ROI-Laplacians.csv")
)  # TIDY
# Wide
# laplacians.pivot_table("deg", "id", ["freq", "roi"], aggfunc="first")
