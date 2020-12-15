"""Array and matrix operations module."""


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
