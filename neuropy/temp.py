"""Temporary module for NeuroPy package.

This module is a temporary home for functions in development. As things start
to mature they will be moved to a more appropriate home. In the meantime, this
allows for more rapid iteration/integration of prototpyed functions.

"""


# Import
import numpy as np
import pandas as pd
import matplotlib as mpl
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


def define_targ_pair_map():
    """Define mapping from targets to target pairs."""

    cond_to_pair = {
        'T1T5': 1,
        'T5T1': 1,
        'T2T6': 2,
        'T6T2': 2,
        'T3T7': 3,
        'T7T3': 3,
        'T4T8': 4,
        'T8T4': 4
    }

    pair_to_cond = {
        1: 'T1T5',
        2: 'T2T6',
        3: 'T3T7',
        4: 'T4T8'
    }

    return cond_to_pair, pair_to_cond


def get_targ_pairs(start_cond, end_cond):
    """Return boolean mask for paired target conditions.

    This function returns a mask indicating which start/end target conditions
    in the provided lists come from a paired set.  For example, the entry for a
    a target condition 'T1 -> T5' will have a corresponding value of True if
    'T5 -> T1' also exists in the data.

    Inputs:
    start_cond - List of strings indicating starting conditions
    end_cond - List of strings indicating ending conditions

    Returns:
    mask - List of boolean indicating whether each of the input elements is
            part of a pair
    """
    
    # Define target pair and flipped target pair conditions
    cond = [''.join([s, e]) for s, e in zip(start_cond, end_cond)]
    filp_cond = [''.join([e, s]) for s, e in zip(start_cond, end_cond)]
    
    # If the flipped version of a target condition appears in the set of unique
    # target conditions, then the target has a matching condition
    uni_cond = set(cond)
    mask = [True if c in uni_cond else False for c in filp_cond]
    
    return mask


def define_color_map(style='default'):
    """Define map from target condition to color.
    
    Returns a dict containing, where each element is a color for a particular
    target condition.
    """
    if style == 'default':
        col_map = {
            'T1T5': {'dark': '#cc2500', 'light': '#cc9a8f'},
            'T5T1': {'dark': '#0080b3', 'light': '#8fbbcc'},
            'T2T6': {'dark': '#06a600', 'light': '#91cc8f'},
            'T6T2': {'dark': '#9b00d9', 'light': '#c698d9'},
            'T3T7': {'dark': '#cc8800', 'light': '#ccb88f'},
            'T7T3': {'dark': '#0044cc', 'light': '#8fa3cc'},
            'T4T8': {'dark': '#77cc00', 'light': '#b3cc8f'},
            'T8T4': {'dark': '#cc0066', 'light': '#cc8fad'}
        }
    elif style == 'circular':
        col_map = {
            'T1T5': {'dark': '#bf0000', 'light': '#bf6060'},
            'T2T6': {'dark': '#bf8f00', 'light': '#bfa760'},
            'T3T7': {'dark': '#60bf00', 'light': '#8fbf60'},
            'T4T8': {'dark': '#00bf30', 'light': '#60bf78'},
            'T5T1': {'dark': '#00bfbf', 'light': '#60bfbf'},
            'T6T2': {'dark': '#0030bf', 'light': '#6078bf'},
            'T7T3': {'dark': '#6000bf', 'light': '#8f60bf'},
            'T8T4': {'dark': '#bf008f', 'light': '#bf60a7'}
        }
    
    return col_map


def plot_traj(traj, cond, col_map,
              onset=None,
              offset=None,
              axh=[],
              col_mode='dark',
              line_width=1,
              marker_size=7,
              mode='2d'):
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
    if onset is None and offset is None:
        truncate_flag = False
    elif onset is not None and offset is not None:
        truncate_flag = True
    else:
        msg = 'Both onset and offset arguments must be specified or empty.'
        raise(Exception(msg))

    # For now, use a for loop:
    for i in range(traj.shape[0]):
        temp_traj = traj.iloc[i]
        # Truncate trajectory if necessary
        if truncate_flag:
            idx = range(onset.iloc[i], offset.iloc[i] + 1)
            temp_traj = temp_traj[:, idx]

        # Plot
        plot_single_traj(temp_traj,
                         cond.iloc[i], col_map,
                         axh=axh,
                         col_mode=col_mode,
                         line_width=line_width,
                         marker_size=marker_size,
                         mode=mode)
    
    return None
    

def plot_single_traj(traj, cond, color_map,
                     axh=[],
                     col_mode='dark',
                     line_width=1,
                     marker_size=10,
                     mode='2d'):
    """Plot a single trajectory"""

    # If an axis handle is not provided, then use the current axis
    if not axh:
        axh = plt.gca()

    # Plot trajectory
    if mode == '2d':
        axh.plot(
            traj[0, :], traj[1, :],
            color=color_map[cond][col_mode],
            linewidth=line_width
        )
        # Plot start point
        axh.plot(
            traj[0, 0], traj[1, 0],
            color=color_map[cond][col_mode],
            marker='.',
            markersize=marker_size,
            markeredgecolor=None,
            markerfacecolor=color_map[cond][col_mode]
        )
        # Plot end point
        axh.plot(
            traj[0, -1], traj[1, -1],
            color=color_map[cond][col_mode],
            marker='o',
            markersize=marker_size,
            markeredgecolor='k',
            markerfacecolor=color_map[cond][col_mode]
        )
    elif mode == '3d':
        axh.plot(
            traj[0, :], traj[1, :], traj[2, :],
            color=color_map[cond][col_mode],
            linewidth=line_width
        )
        # Plot start point
        axh.plot(
            [traj[0, 0]], [traj[1, 0]],
            zs=[traj[2, 0]],
            color=color_map[cond][col_mode],
            marker='.',
            markersize=marker_size,
            markeredgecolor=None,
            markerfacecolor=color_map[cond][col_mode]
        )
        # Plot end point
        axh.plot(
            [traj[0, -1]], [traj[1, -1]], [traj[2, -1]],
            color=color_map[cond][col_mode],
            marker='o',
            markersize=marker_size,
            markeredgecolor='k',
            markerfacecolor=color_map[cond][col_mode]
        )
    
    return None


def spatial_average(traj, max_iter=50):
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
    # TODO: figure out if this is needed
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

    def __int__(self):
        self.X_fit = None
        self.dX_fit = None
        self.nX_fit = None
        self.delta = None
        self.max_dist = None
        self.grid = None
        self.n_grid = None

        return None

    def fit(self, traj, delta, center, max_dist):
        """Fit flow field to data.

        Arguments:
            traj -- Data used to fit the flow field (Series of trajectories)
            delta -- Size of each voxel
            max_dist -- Furthest grid distance
        """
        # Get data
        X, dX, grid, n_grid = get_flow_data(traj, delta, center, max_dist)

        # Iterate over voxels in grid
        X_fit = np.full((n_grid, n_grid, 2), np.nan)  # Voxel center
        dX_fit = np.full((n_grid, n_grid, 2), np.nan)  # Computed flow
        nX = np.zeros((n_grid, n_grid))  # Number of points per voxel
        for i in range(n_grid):  # Iterate over x-dimension
            for j in range(n_grid):  # Iterate over y-dimension
                # Find all points that lie in the current voxel. Currently do
                # this explicitly for the two dimensions, but this could be
                # extended to other dimensions eventually.
                x_1_lim = grid['x'][[i, i+1]]
                x_2_lim = grid['y'][[j, j+1]]
                x_1_mask = np.logical_and(
                    X[0, :] >= x_1_lim[0],
                    X[0, :] < x_1_lim[1])
                x_2_mask = np.logical_and(
                    X[1, :] >= x_2_lim[0],
                    X[1, :] < x_2_lim[1])
                x_mask = np.logical_and(x_1_mask, x_2_mask)  # Could use np.all
                dX_voxel = dX[:, x_mask]

                # Average over points. Only do this if there is at least one
                # point in the voxel
                nX[i, j] = x_mask.sum()
                if nX[i, j] > 0:
                    dX_fit[i, j, :] = dX_voxel.mean(axis=1)

                # Calculate the center of the voxel. The calculated flow vector
                # is associated with this point.
                X_fit[i, j, :] = np.array([grid['x'][i],
                                           grid['y'][j]]) + delta/2

        # Update object with parameters and fit results
        self.X_fit = X_fit
        self.dX_fit = dX_fit
        self.nX_fit = nX
        self.delta = delta
        self.max_dist = max_dist
        self.grid = grid
        self.n_grid = n_grid

        return None

    def plot(self, min_n=1, color='k', axh=[]):
        """Plot flow field."""

        import matplotlib.pyplot as plt

        # Check to see if an axis handle was provided. If so, use this when
        # plotting. If not, get the current axis managed by PyPlot.
        if not axh:
            axh = plt.gca()

        # Unpack data for plotting
        n = self.nX_fit.reshape((1, -1))
        mask = n >= min_n
        X = self.X_fit[:, :, 0].reshape((1, -1))
        Y = self.X_fit[:, :, 1].reshape((1, -1))
        U = self.dX_fit[:, :, 0].reshape((1, -1))
        V = self.dX_fit[:, :, 1].reshape((1, -1))

        # Plot
        axh.quiver(X[mask], Y[mask], U[mask], V[mask], color=color)

        return None


class GaussianFlowField(FlowField):
    """Flow field fit with Gaussian smoothing.

    This class implements fitting flow fields to data using a smoothing-based
    approach. In short, the flow field is 'fit' by taking a weighted average of
    all points for each location where the flow is to be evaluated. Weighting
    is based on the probability that a given point is from a Gaussian
    distribution with a mean at the evaluation points and a variance that is
    fit to the data.

    To find the variance of the distribution, we calculate the variance of the
    magnitude of velocity across data points. This provides a distribution of
    how close neighboring points are in time. [Note 2020.04.29] -- this is
    probably not the best distribution to use. Essentially, what we want to
    know is the probability of a point being in the vicinity of another point.

    The variance term used in the probability distribution can either be fit,
    or it can be specified. The former is best when doing a one-off estimate
    of the flow field. The latter is preferred when comparing different flow
    fields.
    """

    def __init__(self):
        """Init method for GaussianFlowField class."""
        # Call parent init method
        FlowField.__int__(self)

        # Add length constant attribute (not in parent class)
        self.l_const = None

    def fit(self, traj, delta, max_dist, l_const=None):
        """Fit method for Gaussian Flow Field class."""

        # Get data
        X, dX, grid, n_grid = get_flow_data(traj, delta, max_dist)

        # Fit length constant if not specified
        if l_const is None:
            # Calculate length constant here
            D = np.linalg.norm(dX, axis=0, keepdims=True)  # 1 x samp
            l_const = np.mean(D ** 2)  # Average sum of squared distances

        # Pre-compute length constant
        p_coeff = 1 / (np.sqrt(l_const * 2 * np.pi))  # Coefficient

        # Iterate over evaluation points
        X_fit = np.full((n_grid, n_grid, 2), np.nan)
        dX_fit = np.full((n_grid, n_grid, 2), np.nan)
        for i in range(n_grid):
            for j in range(n_grid):
                # Define the current position. Note that the grid entries
                # reflect the edges of the grid, so add half the delta to
                # center them
                x_ij = np.array([[grid[i]], [grid[j]]]) + delta/2

                # Get distance of all points from the current location
                D = np.linalg.norm(X - x_ij, axis=0, keepdims=True)

                # Calculate probability of all points
                pD = p_coeff * np.exp((-1 / 2) * (D ** 2) / l_const)

                # To average, weight each point by its probability, take the
                # sum, and divide by the weighting factors.
                dX_ij = pD * dX
                dx_ij = ((1 / np.sum(pD))
                         * np.sum(dX_ij, axis=1, keepdims=True))

                # Update results matrices
                X_fit[i, j, :] = x_ij[:, 0]  # x_ij is 2D
                dX_fit[i, j, :] = dx_ij[:, 0]  # dx_ij is 2D

        # Add results to object
        self.X_fit = X_fit
        self.dX_fit = dX_fit
        self.l_const = l_const

        # Note - nX is not needed for this class, but set it so that the parent
        # plot method can be used.
        self.nX_fit = np.full((n_grid, n_grid), X.shape[1])

        return None


def get_flow_data(traj, delta, center, max_dist):
    """Get data used for fitting flow fields."""

    # Define grid
    grid = np.arange(0, max_dist, delta)
    grid = np.concatenate([-np.flip(grid[1:]), grid])
    n_grid = grid.shape[0] - 1  # The grid defines the edges
    grid = {'x': grid + center[0, 0], 'y': grid + center[1, 0]}

    # Get velocity and truncated state
    X, dX = get_traj_velocity(traj)  # Output are series
    X = np.concatenate(X.to_numpy(), axis=1)  # Now dim x samp
    dX = np.concatenate(dX.to_numpy(), axis=1)  # Now dim x samp

    return X, dX, grid, n_grid


def remove_non_paired_trials(df):
    """Remove non-paired trials from a dataset.

    This function will remove any trials from the input dataset df that do not
    have a matching pair. A matching pair are trial conditions A->B and B->A.

    """
    # Define target combinations
    start_pos = np.concatenate(df['startPos'].to_numpy())
    end_pos = np.concatenate(df['targPos'].to_numpy())
    targ_comb = np.concatenate([start_pos, end_pos], axis=1)
    uni_targ_comb = np.unique(targ_comb, axis=0)

    # Convert target combinations to trial conditions
    start_cond = get_targ_cond(df['startPos'])
    end_cond = get_targ_cond(df['targPos'])
    targ_cond = [''.join([s, e]) for s, e in zip(start_cond, end_cond)]
    mask = get_targ_pairs(start_cond, end_cond)

    # Remove non-paired targets
    df = df[np.array(mask)]
    targ_cond = [tc for tc, m in zip(targ_cond, mask) if m]

    # Put other target information into a dict for easy access. This is
    # redundant and probably unnecessary, but is being done just in case this
    # information may be useful later on.
    targ_info = {
        'start_pos': start_pos,
        'end_pos': end_pos,
        'targ_comb': targ_comb,
        'uni_targ_comb': uni_targ_comb
    }

    return df, targ_cond, targ_info


def compare_flow_fields(F1, F2, n_min=1):
    """Compare two flow fields.

    """

    # Compute the element-by-element difference
    dX_diff = F1.dX_fit - F2.dX_fit
    dX_diff_mag = np.linalg.norm(dX_diff, axis=2)  # n_grid x n_grid

    # Filter by the number of observations. Find the number of observations for
    # each voxel. Only keep track of magnitude differences that meet this
    # threshold.
    nX_mask_1 = F1.nX_fit < n_min
    nX_mask_2 = F2.nX_fit < n_min
    nX_mask = nX_mask_1 | nX_mask_2
    dX_diff_mag[nX_mask] = np.nan  # Set invalid voxels to NaN

    # In addition to the magnitude matrix, also return just the valid
    # magnitudes, and the number of overlap points
    mag_diff = dX_diff_mag.reshape((-1,))
    mag_diff = mag_diff[np.logical_not(np.isnan(mag_diff))]  # Remove NaN
    n_overlap = np.logical_not(nX_mask).sum()

    # Add results to a dict
    result = {}
    result['diff_grid'] = dX_diff_mag
    result['diff'] = mag_diff
    result['n_overlap'] = n_overlap

    return result


def create_3d_axes():
    """Create 3d axis in new figure."""
    from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots

    # Define figure size in pixels
    ax_size = np.array([500, 500])  # Axis size (pixels)
    x_margin = np.array([100, 100])
    y_margin = np.array([100, 150])
    fw = ax_size[0] + x_margin.sum()
    fh = ax_size[1] + y_margin.sum()
    fig_size = np.array([fw, fh])  # Figure size (pixels)
    fig_size_in = fig_size / mpl.rcParams['figure.dpi']
    fig_hndl = mpl.figure.Figure(figsize=fig_size_in)

    # Define axis position and create
    ax_size_norm = ax_size / fig_size
    x_margin_norm = x_margin / fw
    y_margin_norm = y_margin / fh
    ax_rect = [x_margin_norm[0], y_margin_norm[0],
               ax_size_norm[0], ax_size_norm[1]]
    ax_hndl = fig_hndl.add_axes(ax_rect, projection='3d')

    return fig_hndl, ax_hndl


def plot_fig_4_proj_summary(results):
    """Create summary plot for Fig. 4 analysis

    Create the following plots for each comparison:
    (1) Scatter plot of flow difference (color by target?)
    (2) Scatter plot of voxel overlap (within vs across condition)
    (3) Scatter plot of flow difference vs shared variance
    """

    def plot_scatter(axh, x, y, color, ax_label,
                     plot_unity=True,
                     link_axes=True,
                     n_axis_ticks=3):
        """Plot magnitude difference"""
        axh.scatter(x, y, c=color, alpha=0.5)

        # Axis limits
        if link_axes:
            x_lim = axh.get_xlim()
            y_lim = axh.get_ylim()
            ax_lim = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
            axh.set_xlim(ax_lim)
            axh.set_ylim(ax_lim)

        # Set axis ticks
        x_lim = axh.get_xlim()
        x_tick = np.linspace(x_lim[0], x_lim[1], n_axis_ticks)
        y_lim = axh.get_ylim()
        y_tick = np.linspace(y_lim[0], y_lim[1], n_axis_ticks)
        axh.set_xticks(x_tick)
        axh.set_yticks(y_tick)


        # Plot unity line
        if plot_unity:
            axh.plot(ax_lim, ax_lim, linestyle='--', color='black')

        axh.set_xlabel(ax_label[0])
        axh.set_ylabel(ax_label[1])
        return None

    # Setup figure
    fh, axh = subplot_fixed(2, 4, [300, 300])

    # Get colors for each observation'
    color_map = define_color_map()
    color = [color_map[cond]['dark'] for cond in results['targ_cond']]

    # Plot magnitude difference -- this is the main result that shows whether
    # the intuitive and rotated flow fields are similar.
    col = 0
    plot_scatter(
        axh[0][col],
        results['diff_int'],
        results['diff_int_rot'],
        color,
        ['Flow difference (Int vs Int)', 'Flow difference (Int vs Rot)']
    )
    plot_scatter(
        axh[1][col],
        results['diff_rot'],
        results['diff_int_rot'],
        color,
        ['Flow difference (Rot vs Rot)', 'Flow difference (Int vs Rot)']
    )

    # Plot difference vs shared variance -- this asks whether or not there is a
    # relationship between the amount of shared variance in a given projection
    # and the degree of flow field similarity.
    col += 1
    plot_scatter(
        axh[0][col],
        np.array(results['proj_shared_var']) * 100,
        np.array(results['diff_int_rot']) - np.array(results['diff_int']),
        color,
        ['% shared variance', 'Change in flow difference (Int vs Rot - Int)'],
        plot_unity=False,
        link_axes=False
    )
    plot_scatter(
        axh[1][col],
        np.array(results['proj_shared_var']) * 100,
        np.array(results['diff_int_rot']) - np.array(results['diff_rot']),
        color,
        ['% shared variance', 'Change in flow difference (Int vs Rot - Rot)'],
        plot_unity=False,
        link_axes=False
    )

    # Plot voxel overlap -- this asks whether the number of overlapping voxels
    # is the same between the intuitive and rotated trajectories.
    col += 1
    plot_scatter(
        axh[0][col],
        results['n_overlap_int'],
        results['n_overlap_int_rot'],
        color,
        ['Voxel overlap (Int vs Int)', 'Voxel overlap (Int vs Rot)']
    )
    plot_scatter(
        axh[1][col],
        results['n_overlap_rot'],
        results['n_overlap_int_rot'],
        color,
        ['Voxel overlap (Rot vs Rot)', 'Voxel overlap (Int vs Rot)']
    )

    # Plot number of voxels vs flow field difference -- this tells whether the
    # number of data points being compared influences our results. The thinking
    # here is that comparisons made based on fewer points might be more
    # variable/less reliable than those based on more points.
    col += 1
    ax_str = [
        'Voxel overlap (Int vs Rot)',
        'Change in flow difference (Int vs Rot - Int)'
    ]
    plot_scatter(
        axh[0][col],
        results['n_overlap_int_rot'],
        np.array(results['diff_int_rot']) - np.array(results['diff_int']),
        color,
        ax_str,
        plot_unity=False,
        link_axes=False
    )
    ax_str = [
        'Voxel overlap (Int vs Rot)',
        'Change in flow difference (Int vs Rot - Rot)'
    ]
    plot_scatter(
        axh[1][col],
        results['n_overlap_int_rot'],
        np.array(results['diff_int_rot']) - np.array(results['diff_rot']),
        color,
        ax_str,
        plot_unity=False,
        link_axes=False
    )

    # Add analysis text to figure
    # Set figure title
    title_str = [
        'Subject: {}'.format(results['subject']),
        'Dataset: {}'.format(results['dataset']),
        'Projection mode: {}'.format(results['params']['projection_mode']),
        '# projections: {}'.format(results['params']['n_proj']),
        'Grid spacing: {}'.format(results['params']['grid_delta']),
        'Grid min # overlap: {}'.format(results['params']['grid_n_min']),
    ]
    fh.text(
        0.05, 1 - 0.05,
        '\n'.join(title_str),
        fontsize=12,
        horizontalalignment='left',
        verticalalignment='top'
    )

    return fh
