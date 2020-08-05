"""Module for performing optimization over the stiefel manifold."""

import numpy as np
from scipy import linalg as linalg


def optimize(ObjFn):
    """Perform optimization over the Stiefel manifold."""

    # Parameters
    max_iter = 1000
    max_ls_iter = 500  # Maximum number of line search iterations
    n_restarts = 5  # Number of random restarts to use
    delta_b = 0.9  #
    eps_f = 1e-10

    # Get size of space -- it would be good to get these from the obj. fn.
    x_dim = ObjFn.x_dim  # Size of the data (high-d)
    m_dim = ObjFn.d_dim  # Size of the desired orthonormal space

    # Initialize results
    S = []  # List to store results for each random restart
    for i in range(n_restarts):
        # Initialize M (randomly)
        A = np.random.randn(x_dim, m_dim)
        M = linalg.orth(A)

        # Run gradient descent
        J = []  # Objective function value
        J_terms = []  # Value of objective function terms
        converged_flag = False
        J_iter_prev = np.inf
        for grad_iter in range(max_iter):
            b = 0.1  # Reset step size

            # Step 1: Calculate free gradient
            Z = ObjFn.gradient(M)

            # Step 2: Compute the search direction
            Z = search_dir(-Z, M)

            # Step 3: Line search with retraction
            df = []
            for ls_iter in range(max_ls_iter):
                # Adjust B
                b = b * delta_b

                # Evaluate step
                fM = ObjFn.evaluate(M)
                fR = ObjFn.evaluate(retract(b*Z, M))
                df_iter = fM - fR
                df.append(df_iter)

                # Check for convergence
                if df_iter >= 0:
                    break  # Break out of line search

            # Step 4: Update estimate of M
            M = retract(b*Z, M)

            # Step 5: Check for convergence
            J_iter, J_terms_iter = ObjFn.evaluate(M)
            J.append(J_iter)
            J_terms.append(J_terms_iter)
            dJ = J_iter - J_iter_prev

            # Print convergence status
            if grad_iter % 10 == 0:
                print('Restart {}, Iter {}: J = {:0.3e}, dJ = {:0.3e}'.format(
                    i, grad_iter, J_iter, dJ))

            if abs(dJ) < eps_f:
                converged_flag = True
                break  # Break out of gradient descent

        # Save results for current random restart
        S.append({
            'M': M,
            'J': J,
            'J_terms': J_terms,
            'dJ': dJ,
            'J_final': J[-1],
            'converged': converged_flag,
            'n_iter': grad_iter
        })

    # Find the random restart with the smallest objective function
    J = [s['J_final'] for s in S]  # Final objective function values
    min_idx = np.argmin(J)
    S_final = S[min_idx]

    return S_final


def search_dir(Z, M):
    """Compute Stiefel optimization search direction."""
    x_dim = M.shape[0]
    SK = (1/2) @ (M.T @ Z - Z.T @ M)
    Z = M @ SK + (np.eye(x_dim) - M @ M.T) @ Z
    return Z


def retract(Z, M):
    """Retract onto Stiefel manifold."""
    d_dim = M.shape[1]
    Z = (M + Z) @ (np.eye(d_dim) + Z.T @ Z)**(-1/2)
    return Z


class ObjFn:
    """Objective function class.

    This serves as the base class for objective functions used in the stiefel
    optimization-based approach for finding specific orthogonal projections of
    neural activity.
    """

    def __init__(self, params, data):
        """Initialization function."""
        self.params = params
        self.data = data
        self.x_dim = None
        self.m_dim = None

    def evaluate(self, M):
        """Evaluate objective function."""
        J = 0
        J_terms = [0]
        return J, J_terms

    def gradient(self, M):
        """Evaluate gradient of objective function."""
        dJ = 0
        return dJ


class AsymmetryStandard(ObjFn):
    """Standard (default) asymmetry-defining objective function.

    This class implements a weighted version of the 'standard' objective
    function used for the energy landscape experiments.  It seeks to find a
    projection showing a strong trajectory asymmetry, where the midpoint of the
    A->B and B->A trajectories are maximally different along one axis of the
    projection.

    The weighting parameters (w_mid, w_var, w_start) are used to weight the
    various terms in the objective function.  This was needed b/c the variance
    term can dominate the objective, particularly because it is a second-order
    term and the others are first-order.

    Note that a similar objective function can instead use squared distance to
    minimize this issue.  The 'AsymmetrySquared' class implements this
    objective function.
    """

    def __init__(self, params, data):
        """Initialization function."""

        # Call super method -- this adds the params and data to the object
        super().__init__(params, data)

        # TODO: check parameters structure here

        # Get size of data
        self.x_dim = self.data['mu_A'].shape[0]
        self.m_dim = None  # Currently not needed

    def evaluate(self, M):
        """Evaluate objective function."""

        # Unpack parameters (for clarity)
        w_mid = self.params['w_mid']
        w_var = self.params['w_var']
        w_start = self.params['w_start']

        # Unpack data
        mu_A = self.data['mu_A']
        mu_B = self.data['mu_B']
        mu_AB = self.data['mu_AB']
        mu_BA = self.data['mu_BA']
        sig_AB = self.data['sig_AB']
        sig_BA = self.data['sig_BA']

        # Unpack orthonormal projection
        p_1 = M[:, [0]]
        p_2 = M[:, [1]]

        # --- Compute individual terms in the objective function ---
        # Term 1 -- distance between the centers of the midpoints.  This is
        # positive b/c we want this quantity to be large (meaning that there is
        # a strong asymmetry).
        term_1 = w_mid * p_1.T @ (mu_AB - mu_BA)
        # Term 2 -- sum of variance along p_2 at the midpoint of the
        # trajectories.  This is negative b/c we want this quantity to be small
        # (meaning that the trajectories are consistent at the midpoint).
        term_2 = -w_var * p_1.T @ (sig_AB + sig_BA) @ p_1
        # Term 3 -- distance between the starting points.  This is positive b/c
        # we want this quantity to be large (meaning that the distance
        # between the starting positions is as large as possible.
        term_3 = w_start * p_2.T @ (mu_A - mu_B)

        # Compute overall objective -- this is negative b/c we want to minimize
        J = - (term_1 + term_2 + term_3)
        J_terms = [-term_1, -term_2, -term_3]

        return J, J_terms

    def gradient(self, M):
        """Calculate gradient."""

        # Unpack parameters (for clarity)
        w_mid = self.params['w_mid']
        w_var = self.params['w_var']
        w_start = self.params['w_start']

        # Unpack data
        mu_A = self.data['mu_A']
        mu_B = self.data['mu_B']
        mu_AB = self.data['mu_AB']
        mu_BA = self.data['mu_BA']
        sig_AB = self.data['sig_AB']
        sig_BA = self.data['sig_BA']

        # Unpack orthonormal projection
        p_1 = M[:, [0]]
        p_2 = M[:, [1]]  # NOTE -- this is not used

        # --- Compute derivatives of terms in the objective function ---
        term_1 = w_mid * (mu_AB - mu_BA)
        term_2 = -w_var * 2 * (sig_AB + sig_BA) @ p_1
        term_3 = w_start * (mu_A - mu_B)

        # Combine terms
        d_p_1 = - (term_1 + term_2)
        d_p_2 = -term_3
        dJ = np.concatenate([d_p_1, d_p_2], axis=1)

        return dJ


class AsymmetrySquared(AsymmetryStandard):
    """Distance-squared version of standard asymmetry objective function."""

    def __init__(self, params, data):
        """Initialization function."""

        # Call super method -- this adds the params and data to the object.
        # Additionally, since the data terms used are the same, the size of the
        # data will be set appropriately in the init method of the super class.
        super().__init__(params, data)

    def evaluate(self, M):
        """Evaluate objective function."""
        # TODO: implement this
        J = 0
        J_terms = [0]
        return J, J_terms

    def gradient(self, M):
        """Calculate gradient."""
        # TODO: implement this
        dJ = None
        return dJ
