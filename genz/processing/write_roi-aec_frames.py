#!/usr/bin/env python

"""Compute narrow-band functional connectomes
For each subject, epoch raw .fif resting state into 5 sec. long arbitrary trials data cleaned-up using Autoreject. Do source imaging using ERM SSP,  for each bandwidth (n=7) compute regularized ROI AEC covariance array.
"""

import os
import os.path as op
import time

import numpy as np
import pandas as pd
from genz import defaults, funcs
import mne
from mne.connectivity import envelope_correlation
from mne.minimum_norm import apply_inverse_epochs
from mnefun import get_fsaverage_medial_vertices
from scipy.sparse import csgraph

### locals
new_sfreq = defaults.new_sfreq
lims = [75, 85, 95]
medial_verts = get_fsaverage_medial_vertices()
subjects_dir = defaults.subjects_dir
n_jobs = 4

# TODO refactor to YAML
dfs = []
for ag in [9, 11, 13, 15, 17]:
    fi = op.join(defaults.static, "GenZ_subject_information - %da group.tsv" % ag)
    dfs.append(pd.read_csv(fi, sep="\t", usecols=["Subject Number", "Sex"]))
df = pd.concat(dfs)
df.reset_index(drop=True, inplace=True)
df.columns = ["id", "sex"]
df.sort_values(by="id")
dups = df[df.duplicated("id")].index.values.tolist()
df.drop(df.index[dups], inplace=True)
df.id[:] = [id_.split('_', 1)[1] if isinstance(id_, str) else id_
            for id_ in df.id]
df.drop(df[df.id.isin(defaults.exclude)].index, inplace=True)
df.drop(df[~df.id.isin(defaults.include)].index, inplace=True)
assert len(df) == 106, len(df)
df = df.dropna(how="any")
# df = df.sample(5)
all_rois = mne.read_labels_from_annot(
    "fsaverage", "aparc_sub", "both", subjects_dir=subjects_dir)
rois = [roi for roi in all_rois if not roi.name.startswith("unknown")]
roi_nms = [rr.name for rr in rois]
n = len(roi_nms)
deg_lap = np.zeros((len(df), len(defaults.bands), n))
degree = np.zeros_like(deg_lap)
A_lst = list()
reject = dict(grad=2000e-13, mag=6000e-15)  # same as GenZ repo
src_fs = mne.read_source_spaces(
    op.join(subjects_dir, "fsaverage", "bem", "fsaverage-ico-5-src.fif"))

###start subject loop
for si, ss in enumerate(df.id.values):
    subject = f'genz{ss}'
    del ss
    subj_dir = os.path.join(defaults.megdata, subject)
    inv_fname = os.path.join(
        subj_dir, "inverse",
        f"{subject}-meg-erm-inv.fif")
    raw_fname = os.path.join(
        subj_dir, "sss_pca_fif",
        f"{subject}_rest_01_allclean_fil80_raw_sss.fif"
    )
    src_dir = os.path.join(subj_dir, "source")
    out_fname = os.path.join(src_dir, 'envcorr.h5')
    if op.isfile(out_fname):
        data = mne.externals.h5io.read_hdf5(out_fname)
        degree[si] = data['degree']
        deg_lap[si] = data['deg_lap']
        continue
    print("Loading data for %s" % subject)
    # Load raw
    try:
        raw = mne.io.read_raw_fif(raw_fname)
    except FileNotFoundError as e:
        print(f'    File not found: {raw_fname}')
        continue
    # 0.1 should be okay for 5 sec epochs (envcorr will baseline correct
    # essentially because it's a correlation)
    if raw.info["highpass"] > 0.11:
        print(f"{subject} acquisition HP greater than 0.1 Hz "
              f"({raw.info['highpass']})")
        continue
    if not os.path.exists(src_dir):
        os.mkdir(src_dir)
    raw.load_data()
    # epoch raw into 5 sec trials
    print("    Loading epochs ...", end='')

    # start network loop
    a_lst = dict()
    # epoch raw into 5 sec trials
    events = mne.make_fixed_length_events(raw, duration=5.0)
    tmax = 5.0 - 1. / defaults.new_sfreq
    decim = 4
    # first create originals to get drops
    epochs = mne.Epochs(
        raw, events=events, tmin=0, tmax=tmax, baseline=None,
        reject=reject, preload=True, decim=4)
    assert epochs.info['sfreq'] == defaults.new_sfreq
    print(f" Dropped {len(events) - len(epochs)}/{len(events)} epochs")
    events = epochs.events
    for ix, (kk, vv) in enumerate(defaults.bands.items()):
        print(f'    Processing {"-".join(str(v) for v in vv)} Hz ...', end='')
        t0 = time.time()
        hp, lp = vv
        a_lst[kk] = list()
        # now filter raw and re-epoch
        hp = None if hp == 0 else hp
        print(' Filtering ...', end='')
        raw_use = raw.copy().filter(
            hp, lp, l_trans_bandwidth=1, h_trans_bandwidth=1,
            n_jobs=n_jobs)
        epochs = mne.Epochs(
            raw_use, events=epochs.events, tmin=0, tmax=tmax, baseline=None,
            reject=None, preload=True, decim=decim)
        epochs.apply_hilbert(envelope=False)

        # Compute ROI time series and do envelope correlation
        inv = mne.minimum_norm.read_inverse_operator(inv_fname)
        these_rois = mne.morph_labels(
            all_rois, subject, 'fsaverage', subjects_dir)
        these_rois = [roi for roi in these_rois
                      if not roi.name.startswith("unknown")]
        stcs = apply_inverse_epochs(
            epochs, inv, lambda2=1.0 / 9.0, pick_ori="normal",
            return_generator=True)
        label_ts = mne.extract_label_time_course(
            stcs, these_rois, inv["src"], return_generator=True)

        # compute ROI level envelope power
        print(' Envcorr ...', end='')
        aec = envelope_correlation(label_ts)
        assert aec.shape == (len(rois), len(rois))
        # compute ROI laplacian as per Ginset
        # TODO cite paper
        _, deg_lap[si, ix] = csgraph.laplacian(aec, return_diag=True)

        # compute ROI degree
        degree[si, ix] = mne.connectivity.degree(aec, threshold_prop=0.2)
        # if not np.allclose(deg_lap, degree[si, ix]):
        #     warnings.warn("mne.connectivity.degree NOT equal to laplacian")
        print(f' Completed in {(time.time() - t0) / 60:0.1f} min')
    mne.externals.h5io.write_hdf5(
        out_fname, dict(degree=degree[si], deg_lap=deg_lap[si]))
vertices_to = [s['vertno'] for s in src_fs]

# visualize connectivity on fsaverage
for ix, (kk, vv) in enumerate(defaults.bands.items()):
    grab = np.mean(degree[:, ix], axis=0)  # avg across subjects for this band
    this_stc = mne.labels_to_stc(rois, grab, src=src_fs)
    assert this_stc.data.shape == (20484, 1)
    brain = this_stc.plot(
        subject="fsaverage",
        clim=dict(kind="percent", lims=[75, 85, 95]),
        colormap="gnuplot",
        subjects_dir=subjects_dir,
        views="dorsal",
        hemi="both",
        time_viewer=False,
        show_traces=False,
        smoothing_steps='nearest',
        time_label="%s band" % kk,
    )
    brain.save_image(op.join(defaults.payload, "%s-nxDegree-group-roi.png" % kk))

# write out NxROI laplacian to tidy CSV
foo = funcs.expand_grid({"id": df.id.values, "freq": defaults.bands, "roi": roi_nms})
foo["deg"] = pd.Series(deg_lap.flatten())
foo.to_csv(op.join(defaults.payload, "nxLaplacian-roi.csv"))
# bar = foo.pivot_table("deg", "id", ["freq", "roi"], aggfunc="first").to_csv(
#     op.join(defaults.payload, "nxLaplnsXroi-wide.csv")
# )
