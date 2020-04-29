"""Temporary module for NeuroPy package.

This module is a temporary home for functions in development. As things start
to mature they will be moved to a more appropriate home. In the meantime, this
allows for more rapid iteration/integration of prototpyed functions.

"""


# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_targ_cond(pos):
    """Map target position to unique condition.
    
    Arguments:
    pos -- Array of positions
    
    Returns:
    cond -- List of target condition strings
    
    """
    
    # Define target angles. Need to do this b/c the target positions might be
    # scaled.
    targ_map = {
        -0: 'T1',
        0: 'T1',
        45: 'T2',
        90: 'T3',
        135: 'T4',
        180: 'T5',
        -180: 'T5',
        -135: 'T6',
        -90: 'T7',
        -45: 'T8'
    }
    
    # Iterate over positions, convert to angles, and get associated condition
    # strings
    ang = [np.rad2deg(np.arctan2(p[0, 1], p[0, 0])).astype(int) for p in pos]
    cond = [targ_map.get(k, 'NF') for k in ang]
    
    return cond


def get_targ_pairs(start_cond, end_cond):
    """Return boolean mask for paired target conditions."""
    
    # Define target pair and flipped target pair conditions
    cond = [''.join([s, e]) for s, e in zip(start_cond, end_cond)]
    filp_cond = [''.join([e, s]) for s, e in zip(start_cond, end_cond)]
    
    # If the flipped version of a target condition appears in the set of unique
    # target conditions, then the target has a matching condition
    uni_cond = set(cond)
    mask = [True if c in uni_cond else False for c in filp_cond]
    
    return mask


def define_color_map():
    """Define map from target condition to color.
    
    Returns a dict containing, where each element is a color for a particular
    target condition.
    """
    
    col_map = {
        'T1T5': {'dark': '#cc2500', 'light': '#cc9a8f'},
        'T5T1': {'dark': '#0080b3', 'light': '#8fbbcc'},
        'T2T6': {'dark': '#06a600', 'light': '#91cc8f'},
        'T6T2': {'dark': '#9b00d9', 'light': '#c698d9'},
        'T3T7': {'dark': '#c80', 'light': '#ccb88f'},
        'T7T3': {'dark': '#04c', 'light': '#8fa3cc'},
        'T4T8': {'dark': '#7c0', 'light': '#b3cc8f'},
        'T8T4': {'dark': '#c06', 'light': '#cc8fad'}
    }
    
    return col_map


def plot_traj(traj, cond, col_map,
              onset=[],
              offset=[],
              col_mode='dark',
              line_width=1,
              marker_size=7):
    """Plot trajectories
    
    Plot all trajectories for a set of trials.
    
    Arguments:
    traj -- Pandas series of trajectories (each element is a trial)
    onset_idx -- Pandas series of
    cond -- Pandas series of condition strings
    col_map -- Dict mapping conditions to color values
    
    Keyword arguments:
    col_mode -- 
    line_width --
    marker_size --
    
    """
    # Note -- it might be possible to do this using the Pandas apply() method
    # over the rows in the data frame. However, I was not able to figure out
    # a simple way to pass optional keyword arguments to the anonymous
    # function used with this approach.

    # Set index flag. If the onset and offset indices were passed as arguments,
    # then use these values to truncate the trajectories when plotting.
    if onset == [] and offset == []:
        truncate_flag = False
    elif onset != [] and offset != []:
        truncate_flag = True
    else:
        msg = 'Both onset and offset arguments must be specified or empty.'
        raise(Exception(msg))

    # For now, use a for loop:
    for i in range(traj.shape[0]):
        temp_traj = traj.iloc[i]
        # Truncate trajectory if necessary
        if truncate_flag:
            idx = range(onset_idx.iloc[i], offset_idx.iloc[i] + 1)
            temp_traj = temp_traj[:, idx]

        # Plot
        plot_single_traj(temp_traj,
                         cond.iloc[i], col_map,
                         col_mode=col_mode,
                         line_width=line_width,
                         marker_size=marker_size)
    
    return None
    

def plot_single_traj(traj, cond, color_map,
                     col_mode='dark',
                     line_width=1,
                     marker_size=10):
    """Plot a single trajectory"""
    
    # Plot trajectory
    plt.plot(
        traj[0, :], traj[1, :], 
        color=color_map[cond][col_mode],
        linewidth=line_width             
    )
    # Plot start point
    plt.plot(
        traj[0, 0], traj[1, 0],
        color=color_map[cond][col_mode],
        marker='.',
        markersize=marker_size,
        markeredgecolor=None,
        markerfacecolor=color_map[cond][col_mode]
    )
    # Plot end point
    plt.plot(
        traj[0, -1], traj[1, -1],
        color=color_map[cond][col_mode],
        marker='o',
        markersize=marker_size,
        markeredgecolor='k',
        markerfacecolor=color_map[cond][col_mode]
    )
    
    return None


def spatial_average(
        traj, max_iter=50):
    """Average trajectories spatially.
    """

    import warnings

    # Get associated velocity for each point
    traj_trunc, traj_diff = get_traj_velocity(traj)
    traj_trunc = np.concatenate(traj_trunc.to_numpy(), axis=1)  # dim x samp
    traj_diff = np.concatenate(traj_diff.to_numpy(), axis=1)  # dim x samp

    # When averaging we need to determine which velocities to include in the
    # average for each position. A simple way to do this is just to use an
    # arbitrary threshold, perhaps based on the size of the workspace. However,
    # perhaps a more sensible approach is to use the variance in the velocities
    # directly. To do this, we can convert the velocity vectors to distances
    # and use them to compute the variance. Note that this is making an
    # assumption of equal variance for the two dimensions.
    traj_diff_centered = traj_diff - traj_diff.mean(axis=1, keepdims=True)
    dist = np.linalg.norm(traj_diff_centered, axis=0, keepdims=True)  # 1 x samp
    var_diff = np.mean(dist**2)  # Average sum of squared distances
    p_coeff = 1 / (np.sqrt(var_diff * 2 * np.pi))  # Coefficient

    # Define start and end points of the trajectories. These are used to define
    # where the average trajectory starts and ends.
    start_pos = np.concatenate(
        traj.apply(lambda x: x[:, [0]]).to_numpy(), axis=1)
    start_pos_mean = start_pos.mean(axis=1, keepdims=True)
    end_pos = np.concatenate(
        traj.apply(lambda x: x[:, [-1]]).to_numpy(), axis=1)
    end_pos_mean = end_pos.mean(axis=1, keepdims=True)

    # Calculate end position threshold. This is done by calculating the
    # covariance about the end position for all trajectories and converting
    # this to standard deviation.
    end_var = np.diag(np.cov(end_pos - end_pos.mean(axis=1, keepdims=True)))
    end_radius = np.sqrt(end_var.mean())

    # Initialize average trajectory. Do this because we don't know what the
    # final number of time points will be.
    traj_avg = np.full((traj_trunc.shape[0], max_iter), np.nan)
    traj_avg[:, [0]] = start_pos_mean

    # Iterate
    i = 1  # Start @ 1 b/c this is the index
    curr_pos = traj_avg[:, [0]]
    while i < max_iter:
        # Get all points in the vicinity of the current position.
        dist = np.linalg.norm(traj_trunc - curr_pos, axis=0, keepdims=True)

        # Two options here: (1) define a threshold based on the speed variance,
        # or (2) weight the points. Option (2) is probably preferable here, as
        # it should be more stable for sparse regions of the space.
        p_diff = p_coeff * np.exp((-1/2) * (dist**2) / var_diff)  # weightings

        # To average, weight each point by its probability, take the sum, and
        # divide this by the sum of the weighting factors.
        diff_weighted = p_diff * traj_diff
        diff_avg = ((1/np.sum(p_diff))
                    * np.sum(diff_weighted, axis=1, keepdims=True))

        # Update position
        curr_pos += diff_avg
        traj_avg[:, [i]] = curr_pos
        i += 1

        # Check to see if the trajectory has converged.
        if np.linalg.norm(curr_pos - end_pos_mean) <= end_radius:
            traj_avg = traj_avg[:, 0:i]
            break

    # If the maximum number of iterations was reached, something probably went
    # wrong, so generate a warning
    if i == max_iter:
        msg = ('Maximum number of iterations reached during spatial averaging.'
               + ' Results may be inaccurate.')
        warnings.warn(msg)

    return traj_avg


def get_traj_velocity(traj):
    """Compute velocity for trajectory data.

    This function returns a series of trajectory positions and their
    corresponding velocities.
    """

    # Get velocity for each element in the series. Also truncate the original
    # data, as the velocity cannot be calculated for the last time point of
    # each trajectory.
    traj_diff = traj.apply(np.diff, axis=1)
    traj_trunc = traj.apply(lambda x: x[:, 0:-1])

    return traj_trunc, traj_diff


def find_traj_onset(trial):
    """Find onset and offset of trajectory for a single trial.

    Find the corresponding onset and offset indices.  The onset index should
    be the first time point before the state onset time, since this would
    have been the time where the cursor was inside of the target bound. The
    offset index will be the last time point before the trajectory offset,
    as this will indicate where the criteria was met.

    """

    # Get offset and onset indices
    onset_idx = np.argwhere(trial['decodeTime'][0, :] <
                            trial['stateOnset'])[-1, 0]
    offset_idx = np.argwhere(trial['decodeTime'][0, :] <
                             trial['stateOffset'])[-1, 0]

    # Transform output to series
    traj_idx = pd.Series([onset_idx, offset_idx],
                         index=['trajOnset', 'trajOffset'])

    return traj_idx


class FlowField:
    """Base flow field class."""

    # Methods:
    # - Init
    # - Fit
    # - Plot
    def __int__(self):
        self.X_fit = None
        self.dX_fit = None
        self.delta = None
        self.max_dist = None

        return None

    def fit(self, traj, delta, max_dist):
        """Fit flow field to data.

        Arguments:
            traj -- Data used to fit the flow field (Series of trajectories)
            delta -- Size of each voxel
            max_dist -- Furthest grid distance
        """
        # Define grid
        grid = np.arange(0, max_dist, delta)
        grid = np.concatenate([-np.flip(grid[1:]), x])
        n_grid = grid.shape[0]

        # Get velocity and truncated state
        X, dX = get_traj_velocity(traj)

        # Iterate over voxels in grid
        X_fit = np.full((n_grid, n_grid, 2), np.nan)  # Voxel center
        dX_fit = np.full((n_grid, n_grid, 2), np.nan)  # Computed flow
        nX = np.zeros((n_grid, n_grid))  # Number of points per voxel
        for i in range(n_grid):
            for j in range(n_grid):
                # Find all points that lie in the current voxel

                # Average over points
                dX_fit[i, j, :] = []
                nX[i, j] = []

                # Calculate the center of the voxel
                X_fit[i, j, :] = []

        # Update object with parameters and fit results
        self.X_fit = X_fit
        self.dX_fit = dX_fit
        self.delta = delta
        self.max_dist = max_dist

        return None

    def plot(self):
        """Plot flow field."""
        return None

