"""Validation module for Energy Landscape analysis.

"""

# Import

import numpy as np


def redecode_cursor_pos(spk_cts, pos_online, params, verbose=True):
    """Re-decode cursor positions and compare to online results.

    This function is used to validate that GPFA trajectories extracted offline
    are the same as those used online during experiments.

    Arguments:
    spk_cts - Pandas series containing spike count data
    pos_online - Pandas series containing positions decoded online
    params - Dict with decoding parameters used online

    Returns:
    dif - Array of absolute errors (n_dim * total # of timesteps)
    """

    # Import
    from neuropy.analysis import gpfa

    # Iterate over trials and decode cursor positions
    def decode(Y, params):
        """Decoding cursor position from spike counts."""
        # Extract GPFA trajectories, return decoded position
        U = gpfa.extract_causal_traj(Y, params)
        return params['W'] @ U + params['c']

    # Redecode position for all trials and get difference with online decode
    pos = spk_cts.apply(decode, args=(params,))
    dif = pos - pos_online
    dif = np.concatenate(dif.array, axis=1)
    dif = np.abs(dif).reshape((1, -1))

    # Print results if desired
    if verbose:
        print('Redecode accuracy: {:.3g} (mean), {:.3g} (max)'.format(
            dif.mean(), dif.max()))
    
    return dif
        


