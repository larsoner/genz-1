# -*- coding: utf-8 -*-

import os.path as op
import numpy as np
from scipy import linalg
import mne


def solver(M, G, n_orient):
    """Dummy solver

    It just runs L2 penalized regression and keep the 10 strongest locations

    Parameters
    ----------
    M : array, shape (n_channels, n_times)
        The whitened data.
    G : array, shape (n_channels, n_dipoles)
        The gain matrix a.k.a. the forward operator. The number of locations
        is n_dipoles / n_orient. n_orient will be 1 for a fixed orientation
        constraint or 3 when using a free orientation model.
    n_orient : int
        Can be 1 or 3 depending if one works with fixed or free orientations.
        If n_orient is 3, then ``G[:, 2::3]`` corresponds to the dipoles that
        are normal to the cortex.

    Returns
    -------
    X : array, (n_active_dipoles, n_times)
        The time series of the dipoles in the active set.
    active_set : array (n_dipoles)
        Array of bool. Entry j is True if dipole j is in the active set.
        We have ``X_full[active_set] == X`` where X_full is the full X matrix
        such that ``M = G X_full``.
    """
    K = linalg.solve(np.dot(G, G.T) + 1e15 * np.eye(G.shape[0]), G).T
    K /= np.linalg.norm(K, axis=1)[:, None]
    X = np.dot(K, M)

    indices = np.argsort(np.sum(X ** 2, axis=1))[-10:]  # Maybe return n ICA components # noqa
    active_set = np.zeros(G.shape[1], dtype=bool)
    for idx in indices:
        idx -= idx % n_orient
        active_set[idx:idx + n_orient] = True
    X = X[active_set]
    return X, active_set


def apply_solver(solver, evoked, forward, noise_cov, loose=0.2, depth=0.8):
    """Function to call a custom solver on evoked data

    This function does all the necessary computation:

    - to select the channels in the forward given the available ones in
      the data
    - to take into account the noise covariance and do the spatial whitening
    - to apply loose orientation constraint as MNE solvers
    - to apply a weigthing of the columns of the forward operator as in the
      weighted Minimum Norm formulation in order to limit the problem
      of depth bias.

    Parameters
    ----------
    solver : callable
        The solver takes 3 parameters: data M, gain matrix G, number of
        dipoles orientations per location (1 or 3). A solver shall return
        2 variables: X which contains the time series of the active dipoles
        and an active set which is a boolean mask to specify what dipoles are
        present in X.
    evoked : instance of mne.Evoked
        The evoked data
    forward : instance of Forward
        The forward solution.
    noise_cov : instance of Covariance
        The noise covariance.
    loose : float in [0, 1] | 'auto'
        Value that weights the source variances of the dipole components
        that are parallel (tangential) to the cortical surface. If loose
        is 0 then the solution is computed with fixed orientation.
        If loose is 1, it corresponds to free orientations.
        The default value ('auto') is set to 0.2 for surface-oriented source
        space and set to 1.0 for volumic or discrete source space.
    depth : None | float in [0, 1]
        Depth weighting coefficients. If None, no depth weighting is performed.

    Returns
    -------
    stc : instance of SourceEstimate
        The source estimates.
    """
    # Import the necessary private functions
    from mne.inverse_sparse.mxne_inverse import \
        (_prepare_gain, _check_loose_forward, is_fixed_orient,
         _reapply_source_weighting, _make_sparse_stc)

    all_ch_names = evoked.ch_names

    loose, forward = _check_loose_forward(loose, forward)

    # put the forward solution in fixed orientation if it's not already
    if loose == 0. and not is_fixed_orient(forward):
        forward = mne.convert_forward_solution(
            forward, surf_ori=True, force_fixed=True, copy=True, use_cps=True)

    # Handle depth weighting and whitening (here is no weights)
    gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, evoked.info, noise_cov, pca=False, depth=depth,
        loose=loose, weights=None, weights_min=None)

    # Select channels of interest
    sel = [all_ch_names.index(name) for name in gain_info['ch_names']]
    M = evoked.data[sel]

    # Whiten data
    M = np.dot(whitener, M)

    n_orient = 1 if is_fixed_orient(forward) else 3
    X, active_set = solver(M, gain, n_orient)
    X = _reapply_source_weighting(X, source_weighting, active_set, n_orient)

    stc = _make_sparse_stc(X, active_set, forward, tmin=evoked.times[0],
                           tstep=1. / evoked.info['sfreq'])

    return stc


# Set parameters
data_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample')
raw_fname = op.join(data_path, 'sample_audvis_filt-0-40_raw.fif')
event_fname = op.join(data_path, 'sample_audvis_filt-0-40_raw-eve.fif')
fname_fwd = op.join(data_path, 'sample_audvis-meg-oct-6-fwd.fif')
subjects_dir = op.join(data_path, 'subjects')

# Read raw data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, 30, fir_design='firwin')
events = mne.read_events(event_fname)

# Read epochs
event_id = dict(aud_r=1)
tmin, tmax = -0.2, 0.5
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True,
                       exclude=[])
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=(None, 0),
                    reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))

noise_cov = mne.compute_covariance(
    epochs, tmax=0., method=['shrunk', 'empirical'])

fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw.info)

# Compute the evoked response
evoked = epochs.average()
evoked.plot()
evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type='mag')
evoked.plot_white(noise_cov)  # Show whitening
info = evoked.info

# Read the forward solution and compute the inverse operator
fwd = mne.read_forward_solution(fname_fwd)
fwd = mne.convert_forward_solution(fwd, surf_ori=True)

# Apply your custom solver
loose, depth = 0.2, 0.8  # corresponds to loose orientation
stc = apply_solver(solver, evoked, fwd, noise_cov, loose, depth)

brain = stc.plot(subject='sample', hemi='rh', subjects_dir=subjects_dir,
                 initial_time=0.1, time_unit='s')
brain.show_view('lateral')
