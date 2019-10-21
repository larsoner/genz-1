"""Process GenZ measures using sklearn."""

import os.path as op
import numpy as np
from scipy.stats import spearmanr
import time
from h5io import read_hdf5, write_hdf5
import mne
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, FastICA  # noqa
from sklearn.preprocessing import PolynomialFeatures  # noqa
from sklearn.preprocessing import StandardScaler  # noqa

from sklearn.svm import SVR, SVC, LinearSVC  # noqa
from sklearn.linear_model import LogisticRegression  # noqa
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier  # noqa
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score  # noqa
from sklearn.pipeline import make_pipeline


ages = (9, 11, 13, 15, 17)
freq_idx = [4, 5]  # np.arange(6)  # use all freq ranges
measures = ('betweeness', 'closeness', 'clustering', 'hubness', 'triangles')
n_permutations = 0
scoring = 'balanced_accuracy'
n_jobs = 4
plot_brains = False
plot_correlations = True

freqs = ['DC', 'delta', 'theta', 'alpha', 'beta', 'gamma']
labels = mne.read_labels_from_annot('fsaverage', 'aparc_sub')
labels = [label for label in labels if 'unknown' not in label.name]

###############################################################################
# Load data
# ---------

X, y = [list() for _ in range(len(ages))], list()
for ai, age in enumerate(ages):
    shape = None
    for mi, measure in enumerate(measures):
        fast_fname = 'genz_%s_%s_fast.h5' % (age, measure)
        if not op.isfile(fast_fname):
            print('Converting %s measure %s' % (age, measure))
            data = read_hdf5('genz_%s_%s.h5' % (age, measure))
            data = data['data_vars'][measure]['data']
            data = np.array(data)
            assert data.dtype == np.float
            write_hdf5(fast_fname, data)
        data = read_hdf5(fast_fname)
        if shape is None:
            shape = data.shape
            assert shape[-1] == 2
        assert data.shape == shape
        assert data.ndim == 4
        data = data[freq_idx]  # only use these freqs
        # deal with reordering (undo it to restore original order)
        order = np.argsort(data[:, :, :, 0], axis=-1)
        data = data[..., 1]
        for ii in range(data.shape[0]):
            for jj in range(data.shape[1]):
                data[ii, jj] = data[ii, jj, order[ii, jj]]
        # put in subject, freq, roi order
        data = data.transpose(1, 0, 2)
        data = np.reshape(data, (len(data), -1))  # subjects, ...
        if mi == 0:
            y.extend([age] * len(data))
        X[ai].append(data)
    X[ai] = np.concatenate(X[ai], axis=-1)
y = np.array(y, float)
X = np.concatenate(X, axis=0)
print(X.shape, y.shape)

clf = make_pipeline(
    # PCA(0.9, whiten=True),
    # StandardScaler(),
    # PolynomialFeatures(),
    LogisticRegression(C=1., random_state=0, max_iter=1000, solver='lbfgs', multi_class='multinomial'),  # noqa
    # LogisticRegression(C=50. / len(X), multi_class='multinomial', random_state=0, penalty='l1', solver='saga', tol=0.01),  # noqa
    # SVR(degree=1),
    # RandomForestRegressor(random_state=0, n_estimators=len(ages)),
    # SVC(),
    # LinearSVC(),
    # GradientBoostingClassifier(),
)


###############################################################################
# Do some plotting
# ----------------
# This is only correct for LogisticRegression without a PCA step!
# We can probably figure it out for others but fortunately LR works the best.

if plot_brains:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    t0 = time.time()
    perf, perfs, p = permutation_test_score(
        clf, X, y, scoring=scoring, cv=cv,
        n_permutations=n_permutations, n_jobs=n_jobs, random_state=0,
        verbose=1)
    print('Completed in %0.1f min:' % ((time.time() - t0) / 60.,))
    print('Accuracy:       %0.3f%%' % (100 * perf,))
    print('Significance:   %f' % (p,))
    reg = clf.steps[1][1]
    reg.fit(clf.steps[0][1].fit_transform(X), y=y)
    coef = reg.coef_
    coef.shape = (len(ages), len(measures), len(freq_idx), 448)
    coef = np.abs(coef).max(0).max(0)
    assert len(labels) == coef.shape[-1]
    for ii, fi in enumerate(freq_idx):
        stc = mne.labels_to_stc(labels, coef[ii])
        brain = stc.plot(
            views=['lat', 'med'], hemi='split', smoothing_steps=1,
            clim=dict(kind='value', lims=[0.012, 0.024, 0.036]),
            colormap='viridis', colorbar=False, size=(800, 800),
            time_label=freqs[fi])
        brain.save_image('maxs_%s.png' % (freqs[fi],))
        brain.close()

if plot_correlations:
    # naively compute correlation coefficients and plot a few
    this_X = X - X.mean(0)
    this_X /= np.linalg.norm(this_X, axis=0)
    this_y = y - y.mean()
    this_y /= np.linalg.norm(this_y)
    # Pearson R is bad:
    # corrs = np.dot(y, this_X) / len(y)
    # assert (corrs <= 1.).all() and (corrs >= -1).all()
    # Use Spearman R instead (more robust anyway):
    corrs = spearmanr(this_X, y)[0][-1][:-1]
    shape = (len(measures), len(freq_idx), len(labels))
    assert corrs.shape == (np.prod(shape),)
    idx = np.argmax(np.abs(corrs))
    mi, fi, li = np.unravel_index(idx, shape)
    print('ρ=%0.3f for %r of %r in %r'
          % (corrs[idx], measures[mi], freqs[freq_idx[fi]], labels[li].name))
    ylabel = '%s in %s\n%s' % (measures[mi].capitalize(), freqs[freq_idx[fi]],
                               labels[li].name)
    data = X[:, idx]
    data = [data[y == age] for age in ages]
    fig, ax = plt.subplots(1, figsize=(5, 3))
    ax.violinplot(data, ages, showmeans=True, showextrema=False)
    ax.set(xlabel='Age (yr)', ylabel=ylabel)
    for key in ('top', 'right'):
        ax.spines[key].set_visible(False)
    fig.tight_layout()
