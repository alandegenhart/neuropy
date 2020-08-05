"""GPFA analysis module.

This module contains various functionality for analysis using Gaussian
Process Factor Analysis (GPFA).

"""

# Import
import numpy as np


def extract_causal_traj(Y, params):
    """Extract GPFA trajectories from binned spike counts using estimated
    parameters.

    Agruments:
    :Y -- matrix of spike counts (units x time)
    :params -- GPFA parameters

    Adapted from 'run_causal_GPFA_offline.m' by Alan Degenhart.

    """

    t1 = np.zeros((params['xDim'], params['nBins']))  # Latent history
    
    # Iterate over time points
    n_samp = Y.shape[1]
    U = np.full((params['xDim'], n_samp), np.nan)
    for t in range(n_samp):
        # Shift the latents
        t1[:, :-1] = t1[:, 1:]

        # Get spike counts for current time step, subtract off mean, convert to
        # latents
        y_t = Y[:, [t]]
        dif = y_t - params['d']
        t1[:, [-1]] = params['CRinv'] @ dif  # latents * time

        # Reshape to a column vector. Note that MATLAB and Numpy default
        # conventions for the 'reshape' command are different. 
        term_1 = t1.reshape((params['xDim'] * params['nBins'], 1), order='F')

        # Apply smoothing
        U[:, [t]] = params['M'] @ term_1

    return U


def orthogonalize(C):
    """Orthogonalize GPFA loadings.
    
    This function returns an orthonormalized basis for the loading matrix C, as
    well as the transformation matrix from non-orthonormalized latents to
    orthonormalized latents.

    Parameters:

    Returns:
    C_orth : array
    T : array
    
    Adapted from orthogonalize.m by Byron Yu
    """
    x_dim = C.shape[1]
    if x_dim == 1:
        # If there is only a single latent dimension, just scale C to be a
        # unit vector
        T = np.sqrt(C.T @ C)
        C_orth = C / T
        s = []  # Not currently defined
        VH = []  # Not currently defined
    else:
        # Perform SVD. Note that in standard notation, X = USV', so here the
        # matrix VH := V' (technically the complex transpose).
        C_orth, s, VH = np.linalg.svd(C, full_matrices=False)
        T = np.diag(s) @ VH  # Transformation matrix

    return C_orth, T, s, VH
