"""AIBS dimensionality reduction (dimred) module

This module contains functionality for performing various dimensionality
reduction, including implementations of methods such as Factor Analysis and
utilities for working matrix spaces (e.g., orthogonalization, subspace overlap)
"""
# Import
import numpy as np
import neuropy.mathutil
import scipy.linalg as splinalg


class FactorAnalysis:
    """Factor Analysis (FA)

    This class implements Factor analysis using EM to fit the parameters of the
    latent model. This implementation has been adapted from fastfa.m by Byron
    Yu and has been made consistent with the scikit-learn decomposition module.
    """

    def __init__(self, n_components=10, typ='fa', tol=1e-8, cyc=int(1e8),
                 min_var_frac=0.01, verbose=False, random=False):
        self.n_components = n_components
        self.typ = typ.lower()
        self.tol = tol
        self.cyc = cyc
        self.min_var_frac = min_var_frac
        self.verbose = verbose
        self.random = random

        # Parameter placeholders
        self.L = np.array([])
        self.Ph = np.array([])
        self.d = np.array([])

        # Placeholders for terms generated during fitting
        self.const = 0
        self.I_z = np.array([])
        self.x_dim = []
        self.z_dim = self.n_components

        # Placeholders for results
        self.n_iter = 0
        self.ll_final = 0
        self.ll = np.array([])

    def fit(self, X, y=None):
        """Fit parameters using expectation maximization."""
        from scipy.stats.mstats import gmean

        # Transpose X so the below implementation is consistent with fastfa.m
        X = X.T  # Now x_dim x N

        # Initialize random state. By passing a seed of 0 we are ensuring that
        # repeated runs of this function will return the same initial (random)
        # parameters. It might be worth adding the option to perform different
        # random initialization in the future.
        if self.random:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(0)
        x_dim, N = X.shape

        # Initialize parameters
        cX = np.cov(X, bias=True)
        if np.linalg.matrix_rank(cX) == x_dim:
            scale = np.exp(neuropy.mathutil.logdet(cX) / x_dim)
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
            MM, beta = self.calc_e_step(Ph, L, I_z)
            cX_beta = cX @ beta.T
            EZZ = I_z - beta @ L + beta @ cX_beta  # E[zz']

            # Compute LL. Note that this is slightly different than what was
            # done in the original code. In the case of the fitfa.m function,
            # the covariance cX already includes a normalization constant. In
            # contrast, the sample covariance XcXc in the estep function does
            # not include this normalization, but it has been worked into the
            # LL calculation.
            LLold = LLi
            LLi = self.calc_ll(N, const, MM, cX)
            if self.verbose:
                print(f'EM iteration {i:6d}, LL = {LLi:8.1f}')
            LL.append(LLi)

            # --- M-step ---
            L = cX_beta @ np.linalg.inv(EZZ)
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

        # Add final log-likelihood values. Note that there is a slight
        # discrepancy here between the "final" LL and that which is obtained by
        # re-calculating the LL using the final parameter values. This is
        # because the LL is calculated at the end of the e-step, the parameters
        # are updated in the m-step, and the convergence check happens after the
        # m-step. Thus, the "final" LL value reflects the penultimate parameter
        # update. For now, this will be left as-is, consistent with the original
        # fastfa.m implementation, but in the future it might make more sense to
        # update this so that the final LL value is consistent with the final
        # parameters.
        self.n_iter = i
        self.ll_final = LLi
        self.ll = LL

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
        ll = self.calc_ll(N, self.const, MM, XcXc, normalized=False)

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
        Xc = X - self.d  # Centered version of X
        XcXc = Xc @ Xc.T

        # Calculate posterior mean and covariance
        MM, beta = self.calc_e_step(self.Ph, self.L, self.I_z)
        Z_mean = beta @ Xc  # z_dim x N
        Z_cov = self.I_z - beta @ self.L  # Not returned by sklearn transform method

        # Calculate log likelihood
        ll = self.calc_ll(N, self.const, MM, XcXc, normalized=False)

        return Z_mean.T, Z_cov, ll  # Transpose to keep consistent with scikit-learn conventions

    @staticmethod
    def calc_ll(N, const, MM, S, normalized=True):
        """Helper function to perform LL calculation."""
        term_1 = N * const + 0.5 * N * neuropy.mathutil.logdet(MM)
        term_2 = 0.5 * np.sum(MM * S)
        if normalized:
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
        MM = iPh - iPhL @ np.linalg.inv(I_z + L.T @ iPhL) @ iPhL.T  # C^-1
        beta = L.T @ MM  # z_dim x z_dim
        return MM, beta


def orthogonalize(C):
    """Orthogonalize (loading) matrix.

    This function returns an orthonormalized basis for the loading matrix C, as
    well as the transformation matrix from non-orthonormalized latents to
    orthonormalized latents.

    Parameters:
    C : Loading matrix to orthonormalize (n_features x n_latents)

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
        s = T[0]
        VH = []  # Not currently defined
    else:
        # Perform SVD. Note that in standard notation, X = USV', so here the
        # matrix VH := V' (technically the complex transpose).
        C_orth, s, VH = np.linalg.svd(C, full_matrices=False)
        T = np.diag(s) @ VH  # Transformation matrix

    return C_orth, T, s, VH


def percent_variance_captured(V_1, V_2, S=None):
    """Estimate percent shared variance of one space captured by another.

    Calculates the fraction of the total variance in V_1 that is captured by
    the orthonormal space of V_2.
    """

    # Notation:
    # V: non-orthonormalized space
    # U: orthonormalized space
    U_2, _, _, _ = orthogonalize(V_2)

    # Calculate shared variance/covariance if not specified
    if S is None:
        S = V_1 @ V_1.T

    var_cap = np.trace((U_2 @ U_2.T) @ S @ (U_2 @ U_2.T))
    total_var = np.trace(S)
    p_cap = var_cap / total_var

    return p_cap


def calculate_d_shared(L, threshold=0.95):
    """Calculate dimensionality based on percent shared variance."""

    _, _, s, _ = orthogonalize(L)
    cum_var = np.cumsum(s) / np.sum(s)
    d_shared_mask = cum_var >= threshold
    if d_shared_mask == []:
        print(f'Warning: unexpected cumsum behavior -- no elements are greater than the specified threshold.')
        print(f's = {s}')
        d_shared = 0
    else:
        d_shared = np.argmax(d_shared_mask) + 1
    return d_shared, s


if __name__ == '__main__':
    # Test FA class

    # Define parameters
    x_dim = 100
    z_dim = 10
    N = 100000

    # Generate random parameters
    # For simulating loadings, just use numbers drawn from a standard normal
    # distribution. For Phi, we need positive numbers, so use a gamma
    # distribution with k=2. Use the scale factor to balance the variance of the
    # noise with the shared component.
    rng = np.random.default_rng()
    L = rng.standard_normal(size=(x_dim, z_dim))
    Ph = rng.gamma(shape=4, scale=0.25, size=(x_dim, 1))
    d = rng.standard_normal(size=(x_dim, 1))

    # Check shared variance of latents and display a warning message if any of
    # these elements are small.
    L_orth, T, s, VH = orthogonalize(L)
    if np.any(s < np.mean(Ph)):
        print('Warning, at least one dimension of L has shared variance less than the noise variance.')

    # Generate latent observations. These are zero-mean and unit variance
    Z = rng.multivariate_normal(np.zeros((z_dim, )), np.eye(z_dim), N).T

    # Project into high-D space
    X = L @ Z
    e = np.stack([rng.normal(0, np.sqrt(p), size=(N,)) for p in Ph])  # Noise (zero-mean, variance given by Phi)
    X = (X + np.tile(d, N)) + e

    # Instantiate class and fit parameters
    mdl = FactorAnalysis(n_components=z_dim, verbose=True)
    mdl.fit(X.T)

    # Check post-fitting methods
    ll = mdl.score_samples(X.T)
    Z_predicted = mdl.transform(X.T)
    Z_predicted_estep, Z_cov, ll_estep = mdl.e_step(X.T)

    # Verify that re-computed LL is less than the final LL
    ll_diff = ll - mdl.ll_final
    if ll_diff > 0:
        ll_check = 'PASSED'
    else:
        ll_check = 'FAILED'

    # Verify that Z_predicted and Z_predicted_estep are the same
    z_err = np.sum(Z_predicted - Z_predicted_estep)
    if z_err == 0.0:
        z_check = 'PASSED'
    else:
        z_check = 'FAILED'

    # It appears the fit version of L does not match the true version, but the
    # other parameters (mean d and variance Ph) do. This is likely because the
    # neurons are observed variables, while the latents themselves are
    # unobserved. Thus, it is not guaranteed that the exact same latents will
    # be found (?).
    d_err = np.abs(d - mdl.d)
    d_err_mean = np.mean(d_err)
    d_err_max = np.max(d_err)
    Ph_err = np.abs(Ph - mdl.Ph[:, np.newaxis])
    Ph_err_mean = np.mean(Ph_err)
    Ph_err_max = np.max(Ph_err)

    # To compare loadings, compute percent variance captured
    p_cap_L = percent_variance_captured(mdl.L, L)
    p_cap_random = percent_variance_captured(rng.standard_normal(size=(x_dim, z_dim)), L)

    # Finally check the latents themselves. Because the order of the latents is
    # not guaranteed to be the same, try comparing the orthonormalized latents
    # instead.
    Z_orth = T @ Z
    L_fit_orth, T_fit, s_fit, VH_fit = orthogonalize(mdl.L)
    Z_pred = mdl.transform(X.T).T
    Z_pred_orth = T_fit @ Z_pred

    # Iterate over orthonormalized dimensions and compare orthonormalized
    # latents
    def calc_latent_corr(Z_1, Z_2):
        """Calculate correlation between latent variables"""
        r_orth = []
        for i in range(Z_1.shape[0]):
            R = np.corrcoef(np.stack([Z_1[i, :], Z_2[i, :]]))
            r_orth.append(R[0, 1])  # Want the off-diagonal
        # Since the latents are unique up to a change in sign, there is no
        # distinction between correlations of +1 or -1.
        r_orth = np.abs(r_orth)
        return r_orth

    # Calculate correlation between ground truth latents and those extracted
    # using the fit parameters
    r_orth_fit = calc_latent_corr(Z_orth, Z_pred_orth)
    r_orth_fit_mean = np.mean(r_orth_fit)
    r_orth_fit_min = np.min(r_orth_fit)

    # Calculate correlation between ground truth latents and a second set of
    # random latents
    Z_random = rng.multivariate_normal(np.zeros((z_dim,)), np.eye(z_dim), N).T
    Z_random_orth = T @ Z_random
    r_orth_random = calc_latent_corr(Z_orth, Z_random_orth)
    r_orth_random_mean = np.mean(r_orth_random)
    r_orth_random_max = np.max(r_orth_random)

    print('')
    print('-----------------------------------------')
    print('Factor analysis implementation validation')
    print('-----------------------------------------')
    print('')
    print(f'Parameter d error (mean): {d_err_mean}')
    print(f'Parameter d error (max): {d_err_max}')
    print(f'Parameter Phi error (mean): {Ph_err_mean}')
    print(f'Parameter Phi error (max): {Ph_err_max}')
    print('')
    print(f'Fraction of ground-truth variance captured by fit loadings: {p_cap_L}')
    print(f'Fraction of variance captured by random loadings: {p_cap_random}')
    print('')
    print(f'Reconstruction check: {z_check}')
    print(f'LL check: {ll_check}')
    print('')
    print(f'Orthonormalized latent correlation (mean): {r_orth_fit_mean}')
    print(f'Orthonormalized latent correlation (min): {r_orth_fit_min}')
    print(f'Orthonormalized latent correlation (random, mean): {r_orth_random_mean}')
    print(f'Orthonormalized latent correlation (random, max): {r_orth_random_max}')
