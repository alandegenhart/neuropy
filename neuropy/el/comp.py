"""Computation module for 'el' package.

This module contains various computation functionality used in various energy
landscape analyses.
"""

# Import
import numpy as np
import pandas as pd


def get_asymmetry_features(traj, cond_code):
    """Get set of features defining neural trajectory asymmetries.

    Inputs:
    :traj -- pandas series of neural trajectory matrices
    :cond_code -- associated conditions for each trajectory
    """

    # Get data size
    n_trials = len(traj)
    x_dim = traj.iloc[0].shape[0]

    # The following features need to be calculated:
    # mu_A -- start of A->B trajectory
    # mu_B -- start of B->A trajectory
    # mu_AB -- midpoint of A->B trajectory
    # mu_BA -- midpoint of B->A trajectory
    # sig_AB -- variance at midpoint of A->B trajectory
    # sig_BA -- variance at midpoint of B->A trajectory

    # Define start and end points
    x_start = []
    x_end = []
    for x in traj:
        # Get the first element in the vector
        x_start.append(x[:, [0]])
        x_end.append(x[:, [-1]])

    # Concatenate start and end
    x_start = np.concatenate(x_start, axis=1)
    x_end = np.concatenate(x_end, axis=1)

    # Define masks for target conditions
    uni_cond = set(cond_code)
    cond_mask = [
        [True if cc == uc else False for cc in cond_code]
        for uc in uni_cond
    ]

    # Get average start and end state
    mu_start = []
    mu_end = []
    for cm in cond_mask:
        mu_start.append(x_start[:, cm].mean(axis=1, keepdims=True))
        mu_end.append(x_end[:, cm].mean(axis=1, keepdims=True))

    # Define projection vector between starting positions
    p = mu_start[0] - mu_start[1]
    p = p / np.linalg.norm(p)

    # Iterate over trajectories and project onto target axis
    features = []
    d_p = []
    d_p_start = []
    d_p_end = []
    for x in traj:
        # NOTE -- in the existing code the start and end points were averaged
        # over 2 points.  This has been duplicated here, but might not be
        # necessary.
        d_p_x = x.T @ p  # Project onto target axis
        d_p.append(d_p_x)
        d_p_start.append(d_p_x[0:2].mean())
        d_p_end.append(d_p_x[-2:].mean())

    # Get average starting position along the A-B axis and find closest point
    # for each trial
    d_p_start = np.array(d_p_start)
    p_start = [d_p_start[cm].mean() for cm in cond_mask]
    p_center = np.array(p_start).mean()
    center_idx = [np.argmin(np.abs(dpt - p_center)) for dpt in d_p]

    # Get data at the midpoint for each target condition and calculate mean and
    # covariance
    x_center = [x[:, [cidx]] for x, cidx in zip(traj, center_idx)]
    x_center = np.concatenate(x_center, axis=1)
    mu_center = []
    cov_center = []
    for cm in cond_mask:
        # Calculate mean and covariance
        x_center_cond = x_center[:, cm]
        mu_center.append(x_center_cond.mean(axis=1, keepdims=True))
        cov_center.append(np.cov(x_center_cond))

    # Pack up features.  This will be a dictionary with the various features,
    # where each feature is a list of the feature value for both target
    # conditions.
    features = {
        'cond': cond_code,
        'uni_cond': uni_cond,
        'cond_mask': cond_mask,
        'mu_start': mu_start,
        'mu_end': mu_end,
        'mu_center': mu_center,
        'cov_center': cov_center,
        'x_center': x_center
    }
    return features


