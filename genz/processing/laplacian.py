
from scipy import linalg
from scipy.linalg import sqrtm
from numpy.random import random
from numpy import iscomplexobj
from numpy import trace
from numpy import cov
from scipy.sparse import csgraph
import itertools
import os.path as op
from sklearn.covariance import GraphicalLassoCV, shrunk_covariance

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chord import Chord

import mne
from mne.connectivity import envelope_correlation
from mne.minimum_norm import apply_inverse_epochs, make_inverse_operator
from mne.preprocessing import compute_proj_ecg, compute_proj_eog


def _ht2_p(mu, sigma, n, mu_2, sigma_2, n_2, use_pinv=False):
    """Compute p values using Hotelling T2.

    This will overwrite sigma!

    See:
    - https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution
    - https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Hotellings_One-Sample_T2.pdf
    - https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Hotellings_Two-Sample_T2.pdf
    """  # noqa: E501
    nones = (mu_2 is None, sigma_2 is None, n_2 is None)
    assert all(nones) or not any(nones)  # proper input
    if mu_2 is None:
        # one-sample case
        T2_scale = n
        F_dof = max(n - 3, 1)
    else:
        # two-sample case, make adjustments
        T2_scale = (n * n_2) / (n + n_2)
        F_dof = max(n + n_2 - 4, 1)
        mu = mu - mu_2
        sigma = (((n - 1) * sigma + (n_2 - 1) * sigma_2) /
                 (n + n_2 - 2))
    # Compute the inverses of these covariance matrices:
    if use_pinv:
        sinv = np.linalg.pinv(sigma, rcond=1e-6)
    else:
        try:
            sinv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            sinv = np.linalg.pinv(sigma)
    # Compute the T**2 values:
    T2 = np.einsum('vo...,von,vn...->v...', mu, sinv, mu)
    # Then convert T**2 for p (here, 3) variables and n DOF into F:
    #
    #     F_{p,n-p} = \frac{n-p}{p*(n-1)} * T ** 2
    #
    F = T2  # rename
    F *= T2_scale * (F_dof / float(3 * max(F_dof + 2, 1)))
    # return the associated p values
    return 1 - stats.f.cdf(F, 3, F_dof)

from genz import defaults
import xarray as xr

dfs = list()
for aix, age in enumerate(defaults.ages):
    with xr.open_dataset(op.join(defaults.datadir,
                                 'genz_%s_degree.nc' % age)) as ds:
        df_ = ds.to_dataframe().reset_index()
        df_['age'] = np.ones(len(df_)) * age
        dfs.append(df_)
data_path = mne.datasets.brainstorm.bst_resting.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'bst_resting'
trans = op.join(data_path, 'MEG', 'bst_resting', 'bst_resting-trans.fif')
src = op.join(subjects_dir, subject, 'bem', subject + '-oct-6-src.fif')
bem = op.join(subjects_dir, subject, 'bem', subject + '-5120-bem-sol.fif')
raw_fname = op.join(data_path, 'MEG', 'bst_resting',
                    'subj002_spontaneous_20111102_01_AUX.ds')
raw = mne.io.read_raw_ctf(raw_fname, verbose='error')
raw.crop(0, 60).load_data().pick_types(meg=True, eeg=False).resample(80)
raw.apply_gradient_compensation(3)
projs_ecg, _ = compute_proj_ecg(raw, n_grad=1, n_mag=2)
projs_eog, _ = compute_proj_eog(raw, n_grad=1, n_mag=2, ch_name='MLT31-4407')
raw.info['projs'] += projs_ecg
raw.info['projs'] += projs_eog
raw.apply_proj()
cov = mne.compute_raw_covariance(raw)  # compute before band-pass of interest
raw.filter(14, 30)
events = mne.make_fixed_length_events(raw, duration=5.)
epochs = mne.Epochs(raw, events=events, tmin=0, tmax=5.,
                    baseline=None, reject=dict(mag=8e-13), preload=True)
del raw
src = mne.read_source_spaces(src)
fwd = mne.make_forward_solution(epochs.info, trans, src, bem)
inv = make_inverse_operator(epochs.info, fwd, cov)
del fwd, src

##############################################################################
# Compute label time series and do envelope correlation
# -----------------------------------------------------

labels = mne.read_labels_from_annot(subject, 'aparc_sub',
                                    subjects_dir=subjects_dir)
epochs.apply_hilbert()  # faster to apply in sensor space
stcs = apply_inverse_epochs(epochs, inv, lambda2=1. / 9., pick_ori='normal',
                            return_generator=True)
label_ts = mne.extract_label_time_course(
    stcs, labels, inv['src'], return_generator=True)
corr = envelope_correlation(label_ts, verbose=True)
lala = csgraph.laplacian(corr, normed=False)
# let's plot this matrix
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(corr, cmap='viridis', clim=np.percentile(corr, [5, 95]))
fig.tight_layout()


# #############################################################################
# Estimate the covariance
cov = linalg.inv(corr)
d = np.sqrt(np.diag(cov))
cov /= d
cov /= d[:, np.newaxis]

emp_cov = np.dot(corr.T, corr) / 1
model = GraphicalLassoCV()
model.fit(corr)
cov_ = model.covariance_
corr_ = model.precision_
shrunk = shrunk_covariance(corr)


# #############################################################################
# Plot the covs
plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.02, right=0.98)
covs = [('AEC', corr), ('Laplacian', lala), ('True', cov),
        ('Empirical', emp_cov), ('Shrunk', shrunk),
        ('GraphicalLassoCV', cov_), ]
vmax = emp_cov.max()
for i, (name, this_cov) in enumerate(covs):
    plt.subplot(2, 4, i + 1)
    plt.imshow(this_cov, interpolation='nearest', vmin=-vmax, vmax=vmax,
               cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.title('%s covariance' % name)

# Compute the T**2 values:
mu = lala.mean(axis=0)  # Fréchet??
T2 = np.einsum('i,ij,i->i', mu, cov_, mu)

iu = np.mask_indices(corr.shape[1], np.triu, k=1)
corr_iu = corr[iu]
rois = [ll.name for ll in labels]
r = list(itertools.combinations(rois, 2))
assert len(r) == len(corr_iu)

s_ = pd.Series(r)
ss = pd.Series(corr_iu)
df = pd.concat((s_, ss), axis=1)
print(df.info())
