"""LDA test script

This script uses LDA to find low-dimensional projections of high-d data
generated from random distributions.

"""

# Import libraries
import os
import sys
import pandas as pd
import numpy as np
import scipy.linalg as linalg
import itertools

# Define directory paths and import modules
home_dir = os.path.expanduser('~')
src_dir = os.path.join(home_dir, 'src', 'neuropy')
sys.path.append(src_dir)
import neuropy as neu
from neuropy.analysis import gpfa
import neuropy.temp as tmp
import neuropy.el.ms.fig_4 as f4


def main():

    # Define parameters
    D = 5  # Dimensionality of high-d space
    M = 3  # Number of classes
    N = 100  # Number of samples/class
    sig = 0.1  # High-d variance (shared)
    col = ['r', 'b', 'k']  # Class colors (currently hard-coded)

    # Generate mean and variance parameters
    rng = np.random.default_rng()
    mu = rng.standard_normal((D, M))
    sigma = np.diag(np.ones(D) * sig)

    # Generate random observations
    X = []
    labels = []
    label_col = []
    for i in range(M):
        # Generate samples and matching labels
        X_i = rng.multivariate_normal(mu[:, i], sigma, N)
        labels_i = [i] * N

        # Add to output list
        X.append(X_i.T)
        labels.extend(labels_i)
        label_col.extend([col[i]] * N)
        
    # Concatenate
    X = np.concatenate(X, axis=1)

    # Generate several random projections and plot
    row = range(3)
    col = range(3)
    fh_random, axh = tmp.subplot_fixed(len(row), len(col), [300, 300])
    for r, c in itertools.product(row, col):
        # Generate random 2D projection
        P = linalg.orth(np.random.randn(D, 2))
        X_p = P.T @ X

        # Plot
        axh[r][c].scatter(X_p[0, :], X_p[1, :], c=label_col)

    # Run LDA and plot resultant projection
    D, J = neu.analysis.math.fisher_lda(X, labels)
    X_p = D.T @ X
    fh_lda, axh = tmp.subplot_fixed(1, 1, [300, 300])
    axh[0][0].scatter(X_p[0, :], X_p[1, :], c=label_col)

    # Save plots
    save_dir = os.path.join(
        os.path.expanduser('~'), 'results', 'random', 'lda_test'
    )
    os.makedirs(save_dir, exist_ok=True)
    fig_name = os.path.join(save_dir, 'random_proj.pdf')
    fh_random.suptitle('Random projections')
    fh_random.savefig(fig_name)
    fig_name = os.path.join(save_dir, 'lda_proj.pdf')
    fh_lda.suptitle('LDA projection')
    fh_lda.savefig(fig_name)

    return None


if __name__ == '__main__':
    main()
