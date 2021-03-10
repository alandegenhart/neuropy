"""AIBS math module

This module contains assorted math functions.

"""
# Import
import numpy as np


def ismember_rows(a, b):
    """Return rows of one matrix present in another.
    
    This function finds the rows of a that are equal to b. This function
    requires one of the two input arrays to have a single row.
    """
    
    # Make sure both a and b are two-dimensional. Also check to make sure one
    # of the two arrays has only one row.
    assert(a.ndim == 2 & b.ndim == 2)
    assert(a.shape[0] == 1 or b.shape[0] == 1)
    
    # Check to determine which mode the function is operating. If a is a single
    # row, then the output will be a single boolean indicating if a is a row in
    # b. If b is a row, then the function should return a boolean indicating
    # which rows (if any) in a are equal to b.
    if a.shape[0] == 1:
        mode = 'any'
    else:
        mode = 'all'
    
    # Find matrix of boolean values. Note this uses broadcasting.
    c = np.all(a == b, axis=1)
    
    # If the first argument is a single row, check to see if any of the element
    # s in c are True. If so, a is a row in b.
    if mode == 'any':
        c = np.any(c)
    
    return c


def logdet(A):
    """Computes log(det(A)) where A is positive-definite

    This is faster and more stable than computing log(det(A)) directly.

    Adapted from logdet.m by Tom Minka
    """
    from scipy.linalg import cholesky

    U = cholesky(A)
    y = 2 * np.sum(np.log(np.diag(U)))
    return y


def fisher_lda(X, classes):
    """Fisher linear discriminant analysis.

    Inputs:
    :X -- Design matrix of activity (dim x observations)
    :cond_code -- Condition code for each column in X

    Outputs:
    :D -- Projection vectors
    :J -- Discriminant
    """

    # Import
    from scipy import linalg

    # Transpose input data to row-vector form (rows are now observations)
    X = X.T
    n_dims = X.shape[1]
    unique_classes = set(classes)
    n_classes = len(unique_classes)

    # Initialize within-class scatter
    SS_within = np.zeros([n_dims, n_dims])

    # Iterate over classes
    class_means = np.zeros((n_classes, n_dims))
    for idx, u_cls in enumerate(unique_classes):
        # Get observations for current label
        label_mask = [True if u_cls == cls else False for cls in classes]
        X_class = X[label_mask, :]

        # Calculate mean and covariance
        class_means[idx, :] = X_class.mean(axis=0, keepdims=True)
        SS_within = SS_within + np.cov(X_class.T)

    # Calculate between-class scatter and eigenvalues/vectors
    SS_between = np.cov(class_means.T)
    eig_vals, eig_vecs = linalg.eig(SS_between, SS_within)
    eig_vals = np.real(eig_vals)

    # Get diagonal of eigenvalue matrix and sort
    sort_idx = np.argsort(eig_vals)  # should be ascending order
    sort_idx = np.flip(sort_idx)  # Now in descending order
    eig_vecs = eig_vecs[:, sort_idx]  # Sorted eigenvectors

    # Truncate eigenvectors to number of classes - 1 and calculate discriminant
    n_return_dims = n_classes - 1
    D = eig_vecs[:, 0:n_return_dims]
    eig_within, _ = np.linalg.eig(SS_within)
    eig_between, _ = np.linalg.eig(SS_between)
    J = np.sum(eig_between)/np.sum(eig_within)

    # TODO: would be good to verify that the discriminant is being calculated
    #   correctly -- the way to do this might be to generate data with
    #   different covariance values but the same means and show that the
    #   discriminant value changes accordingly.

    return D, J
