"""Decomposition module

This module contains functionality for performing various matrix decompositions,
including implementations of methods such as Factor Analysis.
"""


import numpy as np
from neuropy.array import logdet


class FactorAnalysis():
    """Factor Analysis (FA)

    This class implements Factor analysis using EM to fit the parameters of the
    latent model. This implementation has been adapted from fastfa.m by Byron
    Yu and has been made consistent with the scikit-learn decomposition module.
    """

    def __init__(self, n_components=10, typ='fa', tol=1e-8, cyc=int(1e8),
                 min_var_frac=0.01, verbose=False):
        self.n_components = n_components
        self.typ = typ
        self.tol = tol
        self.cyc = cyc
        self.min_var_frac = min_var_frac
        self.verbose = verbose

        # Parameter placeholders
        self.L = np.array([])
        self.Ph = np.array([])
        self.d = np.array([])

    def fit(self, X, y=None):
        """Fit parameters using expectation maximization."""
        from scipy.linalg import cholesky as chol
        from scipy.stats.mstats import gmean

        # Transpose X so the below implementation is consistent with fastfa.m
        X = X.T

        # Initialize random state
        # TODO: set random state
        x_dim, N = X.shape

        # Initialize parameters
        cX = np.cov(X.T, bias=True)
        if np.linalg.matrix_rank(cX) == x_dim:
            scale = np.exp(2 * np.sum(np.log(np.diag(chol(cX)))) / x_dim)
        else:
            # cX is not full rank
            print('Warning: data matrix is not full rank.')
            r = np.linalg.matrix_rank(cX)
            e = np.sort(np.linalg.eigvals(cX))[::-1]  # np sort is in ascending order, need to reverse
            scale = gmean(e[0:r])
        rng = np.random.default_rng()
        L = rng.standard_normal(size=(x_dim, self.n_components)) * np.sqrt(scale / self.n_components)
        Ph = np.diag(cX)
        d = np.mean(X, axis=1)

        var_floor = self.min_var_frac * np.diag(cX)

        I_z = np.eye(self.n_components)
        const = -x_dim / 2 * np.log(2 * np.pi)
        LLi = 0
        LL = []

        # Perform EM iterations
        for i in range(self.cyc):
            # --- E-step ---
            pass
            # --- M-step ---

        self.L = L
        self.Ph = Ph
        self.d = d

    def transform(self, X):
        """Extract latents."""

        # Transpose X to be consistent with the features x samples convention
        # used in the fastfa.m implementation
        X = X.T

        Xc = X - self.d  # Centered version of X
        I_z = np.eye(self.n_components)

        iPh = np.diag(1 / self.Ph)
        iPhL = iPh @ self.L
        MM = iPh - iPhL / (I_z + self.L.T @ iPhL) @ iPhL.T  # Note -- need to handle matrix division here
        beta = self.L.T @ MM  # z_dim x z_dim

        Z_mean = beta @ Xc  # z_dim x N

        return Z_mean.T  # Transpose to keep consistent with scikit-learn conventions

    def score_samples(self, X):
        """Extract latent representation of input samples and return log-
        likelihood

        Note that this function assumes the matrix convention used by scikit-
        learn, where X is of the shape (n_observations, n_features).
        """

        # Transpose X to be consistent with the features x samples convention
        # used in the fastfa.m implementation
        X = X.T
        x_dim, N = X.shape

        Xc = X - self.d  # Centered version of X
        XcXc = Xc @ Xc.T
        I_z = np.eye(self.n_components)
        const = x_dim / 2 * np.log(2 * np.pi)

        iPh = np.diag(1/self.Ph)
        iPhL = iPh @ self.L
        MM = iPh - iPhL / (I_z + self.L.T @ iPhL) @ iPhL.T  # Note -- need to handle matrix division here

        # Calculate log likelihood
        ll = N * const + 0.5 * N * logdet(MM) - 0.5 * np.sum(np.sum(MM * XcXc))

        return ll

    def e_step(self, X):
        """Expectation step for FA

        This method computes the posterior mean, covariance, and log-likelihood
        of the input data X. This function should be used when all three of
        these values are needed. If only the posterior mean is needed, the
        transform() method should be used. If only the log-likelihood is needed,
        then the score_samples() method should be used.

        """

        # Transpose X to be consistent with the features x samples convention
        # used in the fastfa.m implementation
        X = X.T
        x_dim, N = X.shape

        Xc = X - self.d  # Centered version of X
        XcXc = Xc @ Xc.T
        I_z = np.eye(self.n_components)
        const = x_dim / 2 * np.log(2 * np.pi)

        iPh = np.diag(1 / self.Ph)
        iPhL = iPh @ self.L
        MM = iPh - iPhL / (I + self.L.T @ iPhL) @ iPhL.T  # Note -- need to handle matrix division here
        beta = self.L.T @ MM  # z_dim x z_dim

        Z_mean = beta @ Xc  # z_dim x N
        Z_cov = I_z - beta @ self.L  # Not returned by sklearn transform method

        # Calculate log likelihood
        ll = N * const + 0.5 * N * logdet(MM) - 0.5 * np.sum(np.sum(MM * XcXc))

        return Z_mean.T, Z_cov, ll  # Transpose to keep consistent with scikit-learn conventions

