"""Decomposition module

This module contains functionality for performing various matrix decompositions,
including implementations of methods such as Factor Analysis.
"""

import numpy as np
from neuropy.array import logdet
from scipy.linalg import cholesky

class FactorAnalysis:
    """Factor Analysis (FA)

    This class implements Factor analysis using EM to fit the parameters of the
    latent model. This implementation has been adapted from fastfa.m by Byron
    Yu and has been made consistent with the scikit-learn decomposition module.
    """

    def __init__(self, n_components=10, typ='fa', tol=1e-8, cyc=int(1e8),
                 min_var_frac=0.01, verbose=False):
        self.n_components = n_components
        self.typ = typ.lower()
        self.tol = tol
        self.cyc = cyc
        self.min_var_frac = min_var_frac
        self.verbose = verbose

        # Parameter placeholders
        self.L = np.array([])
        self.Ph = np.array([])
        self.d = np.array([])
        self.const = 0
        self.I_z = np.array([])
        self.x_dim = []
        self.z_dim = self.n_components

    def fit(self, X, y=None):
        """Fit parameters using expectation maximization."""
        from scipy.stats.mstats import gmean

        # Transpose X so the below implementation is consistent with fastfa.m
        X = X.T  # Now x_dim x N

        # Initialize random state. By passing a seed of 0 we are ensuring that
        # repeated runs of this function will return the same initial (random)
        # parameters. It might be worth adding the option to perform different
        # random initialization in the future.
        rng = np.random.default_rng(0)
        x_dim, N = X.shape

        # Initialize parameters
        cX = np.cov(X, bias=True)
        if np.linalg.matrix_rank(cX) == x_dim:
            scale = np.exp(logdet(cX) / x_dim)
        else:
            # cX is not full rank
            print('Warning: data matrix is not full rank.')
            r = np.linalg.matrix_rank(cX)
            e = np.sort(np.linalg.eigvals(cX))[::-1]  # np sort is in ascending order, need to reverse
            scale = gmean(e[0:r])

        L = rng.standard_normal(size=(x_dim, self.z_dim)) * np.sqrt(scale / self.z_dim)
        Ph = np.diag(cX)
        d = np.mean(X, axis=1, keepdims=True)

        var_floor = self.min_var_frac * np.diag(cX)

        I_z = np.eye(self.n_components)
        const = -x_dim / 2 * np.log(2 * np.pi)
        LLi = 0
        LL = []

        # Perform EM iterations
        for i in range(self.cyc):
            # --- E-step ---
            iPh = np.diag(1 / Ph)
            iPhL = iPh @ L
            MM = iPh - iPhL @ np.linalg.inv(I_z + L.T @ iPhL) @ iPhL.T
            beta = L.T @ MM  # z_dim x z_dim

            cX_beta = cX @ beta.T
            EZZ = I_z - beta @ L + beta @ cX_beta  # Expected cov (?)

            #MM, beta = self.calc_e_step(Ph, L, I_z)

            # Compute LL. Note that this is slightly different than what was
            # done in the original code. In the case of the fitfa.m function,
            # the covariance cX already includes a normalization constant. In
            # contrast, the sample covariance XcXc in the estep function does
            # not include this normalization, but it has been worked into the
            # LL calculation.

            # ldM = np.sum(np.log(np.diag(chol(MM))))  # Original code
            # LLi = N * const + N * ldM - 0.5 * N * np.sum(np.sum(MM * cX))  # Original code
            LLold = LLi
            ldM = np.sum(np.log(np.diag(cholesky(MM))))
            LLi = N * const + N * ldM - 0.5 * N * np.sum(np.sum(MM * cX, axis=1))
            #LLi = self.calc_ll(N, const, MM, cX)
            if self.verbose:
                print(f'EM iteration {i:6d}, LL = {LLi:8.1f}')
            LL.append(LLi)

            # --- M-step ---
            # L = cX_beta / EZZ  NOTE -- original MATLAB operation
            L = cX_beta @ np.linalg.inv(EZZ)
            # LT, _, _, _ = np.linalg.lstsq(EZZ.T, cX_beta.T, rcond=None)  # Equivalent numpy operation
            # L = LT.T
            Ph = np.diag(cX) - np.sum(cX_beta * L, axis=1)

            if self.typ == 'ppca':
                Ph = np.mean(Ph) * np.ones((x_dim, 1))
            elif self.typ == 'fa':
                # Set minimum private variance
                Ph = np.maximum(var_floor, Ph)
            else:
                raise ValueError()

            # Check for convergence
            if i <= 1:
                # Let run for 2 iterations
                LLbase = LLi
            elif LLi < LLold:
                # Display message if LL increases -- this should not happen
                print('VIOLATION: LL decrease observed')
            elif (LLi - LLbase) < ((1 + self.tol) * (LLold - LLbase)):
                # Converged, break
                break

        # Display warning if the private variance floor was used for any dimensions
        if np.any(Ph == var_floor):
            print('Warning: Private variance floor used for one or more dimensions in FA.')

        # Add final parameters to object. Also keep track of some convenience
        # variables (const, I_z) to save from having to re-generate these later
        self.L = L
        self.Ph = Ph
        self.d = d
        self.const = const
        self.I_z = I_z
        self.x_dim = x_dim

    def transform(self, X):
        """Extract latents."""
        # Transpose X to be consistent with the features x samples convention
        # used in the fastfa.m implementation
        X = X.T  # z_dim x samples
        Xc = X - self.d  # Centered version of X

        MM, beta = self.calc_e_step(self.Ph, self.L, self.I_z)
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
        N = X.shape[1]

        Xc = X - self.d  # Centered version of X
        XcXc = Xc @ Xc.T  # Sample covariance (?)

        # Calculate log-likelihood
        MM, beta = self.calc_e_step(self.Ph, self.L, self.I_z)
        ll = self.calc_ll(N, self.const, MM, XcXc)

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
        N = X.shape[1]

        # Center data
        Xc = X - self.d  # Centered version of X
        XcXc = Xc @ Xc.T

        # Calculate posterior mean and covariance
        MM, beta = self.calc_e_step(self.Ph, self.L, self.I_z)
        Z_mean = beta @ Xc  # z_dim x N
        Z_cov = self.I_z - beta @ self.L  # Not returned by sklearn transform method

        # Calculate log likelihood
        ll = self.calc_ll(N, self.const, MM, XcXc, cov_type='unnormalized')

        return Z_mean.T, Z_cov, ll  # Transpose to keep consistent with scikit-learn conventions

    @staticmethod
    def calc_ll(N, const, MM, S, cov_type='normalized'):
        """Helper function to perform LL calculation."""
        term_1 = N * const + 0.5 * N * logdet(MM)
        term_2 = 0.5 * np.sum(MM * S)
        if cov_type == 'normalized':
            term_2 = term_2 * N
        ll = term_1 - term_2
        return ll

    @staticmethod
    def calc_e_step(Ph, L, I_z):
        """Helper function to perform e-step calculations used in multiple
        places in the class
        """
        iPh = np.diag(1 / Ph)
        iPhL = iPh @ L
        # C, _, _, _ = np.linalg.lstsq((I_z + L.T @ iPhL).T, iPhL.T, rcond=None)
        # C = C.T
        # MM = iPh - C @ iPhL.T
        MM = iPh - iPhL @ np.linalg.inv(I_z + L.T @ iPhL) @ iPhL.T
        beta = L.T @ MM  # z_dim x z_dim

        return MM, beta


def orthogonalize(C):
    """Orthogonalize (loading) matrix.

    This function returns an orthonormalized basis for the loading matrix C, as
    well as the transformation matrix from non-orthonormalized latents to
    orthonormalized latents.

    Parameters:

    Returns:
    C_orth : array
    T : array

    Orthonormalized latents can be obtained by computing T @ X, where X is the
    non-orthonormalized latent state (xDim x samples).

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


def percent_variance_captured(L_1, L_2):
    """Estimate percent shared variance of one space captured by another."""


    return p_cap


if __name__ == '__main__':
    # Test FA class

    # Define parameters
    x_dim = 100
    z_dim = 10
    N = 10000

    # Generate random parameters
    rng = np.random.default_rng()
    L = rng.standard_normal(size=(x_dim, z_dim))
    Ph = np.abs(rng.standard_normal(size=(x_dim, 1)))
    d = rng.standard_normal(size=(x_dim, 1))

    # Generate latent observations. These are zero-mean and unit variance
    Z = rng.multivariate_normal(np.zeros((z_dim, )), np.eye(z_dim), N).T

    # Project into high-D space
    X = L @ Z
    e = np.stack([rng.normal(0, np.sqrt(p), size=(N,)) for p in Ph])  # Noise (zero-mean, variance given by Phi)
    X = (X + np.tile(d, N)) + e

    # Instantiate class and fit parameters
    mdl = FactorAnalysis(n_components=z_dim, verbose=True)
    mdl.fit(X.T)

    L

    # Compare fit parameters to ground truth
    L_err = np.sum(np.abs(L - mdl.L))