"""Movie module.

This module provides functionality for processing data during movie
presentation.
"""

# Import - standard modules
import numpy as np
import scipy
import pandas as pd
import os
import sys

from matplotlib import cm

# Import - Allen SDK
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

# Import custom modules
import neuropy as neu
import aibs.dev as dev
import aibs.utilities as util

# Get home directory
home_dir = os.path.expanduser('~')


def plot_single_trial_movie_response(X):
    """Plot single-trial responses to movie presentation for all neurons."""

    # Get maximum event response.  This is used to set the color limits to the
    # same scale
    max_val = X.max()

    # Setup figure.  The easiest thing to do here is probably to create a axis
    # for each neuron.  This will allow for alternating color maps to be used.
    n_neurons = X.shape[0]
    fh, axh = neu.temp.subplot_fixed(
        n_neurons, 1, [1600, 30],
        x_margin=[200, 200],
        y_margin=[200, 300],
        ax_spacing=0
    )

    # Specify valid colormaps
    col_map = ['Blues', 'Oranges']

    # Iterate over neurons
    for n in range(n_neurons):
        # Determine colormap index
        col_idx = n % 2
        x = X[n, :, :].T

        # Plot
        curr_ax = axh[n][0]
        h = curr_ax.imshow(
            x, cmap=cm.get_cmap(col_map[col_idx]),
            aspect='auto',
            interpolation='none')
        curr_ax.set_xticks([])
        curr_ax.set_yticks([])
        curr_ax.set_ylabel(f'{n}')
        h.set_clim([0, max_val])

    return fh


def plot_trial_averaged_movie_response(X, frame_num):
    """Plot trial-avearged response to movie presentation for all neurons."""
    # Imports
    from matplotlib import cm

    # Get maximum event response.  This is used to set the color limits to the
    # same scale
    max_val = X.max()

    # Setup figure.  The easiest thing to do here is probably to create a axis
    # for each neuron.  This will allow for alternating color maps to be used.
    n_neurons = X.shape[0]
    fh, axh = neu.temp.subplot_fixed(
        1, 1, [1000, 1000],
        x_margin=[200, 200],
        y_margin=[200, 300]
    )
    curr_ax = axh[0][0]

    # Specify valid colormaps
    col = ['xkcd:dull blue', 'xkcd:burnt umber']

    # Iterate over neurons
    for n in range(n_neurons):
        # Determine colormap index
        col_idx = n % 2
        x = (X[n, :].T / max_val) + n

        # Plot
        curr_ax.plot(frame_num, x, color=col[col_idx])

    # Format axes
    curr_ax.set_xlabel('Frames')
    curr_ax.set_ylabel('Neuron')
    curr_ax.set_xlim([0, np.max(frame_num)])
    curr_ax.set_ylim([-0.5, n_neurons + 0.5])

    return fh


def convert_movie_stimulus_epochs(stimulus_epochs):
    """Convert raw stimulus epochs for movies returned by the Allen SDK into
    epochs defining the start and end of each movie repeat.

    """
    # Get the minimum and maximum frame numbers and find the occurrence of each
    frame_start = stimulus_epochs.frame.min()
    frame_end = stimulus_epochs.frame.max()
    frame_start_inds = np.argwhere(stimulus_epochs.frame == frame_start).flatten()
    frame_end_inds = np.argwhere(stimulus_epochs.frame == frame_end).flatten()

    movie_epochs = []
    for s_idx, e_idx in zip(frame_start_inds, frame_end_inds):
        movie_epochs.append(
            {
                'repeat': stimulus_epochs.repeat.iloc[s_idx],
                'start': stimulus_epochs.start.iloc[s_idx],
                'end': stimulus_epochs.end.iloc[e_idx]
            }
        )

    # Convert to DataFrame
    movie_epochs = pd.DataFrame(movie_epochs)

    return movie_epochs


def process_saved_movie_responses(
        data,
        bin_size=1,
        smooth=False,
        smooth_win_size=11,
        downsample=False,
        downsample_n=3,
    ):
    """Process a single saved movie response dataset.

    Input: 'data' should be a dictionary with fields 'exp', 'neuron_ids',
    'metadata', and 'response'. 'response' will be an array of size neurons x
    time x repeats. This function performs smoothing, binning, and averaing
    operations on the response array.

    Output: dictionary identical to the input dictionary with the exception of
    the smoothed/binned/averaged responses.
    """
    X = data['response']

    # Perform smoothing and downsampling if desired
    if smooth is True:
        X = dev.smooth_responses(
            X,
            win_size=smooth_win_size,
            ds=downsample,
            ds_n=downsample_n
        )

    # Bin data
    if bin_size > 1:
        # Get start and end of bins. Note that the last bin will be truncated
        bin_start = np.arange(0, X.shape[1], bin_size)
        bin_end = bin_start + bin_size
        bin_end[-1] = X.shape[1]

        # Iterate over bins
        X_binned = []
        for bs, be in zip(bin_start, bin_end):
            X_binned.append(np.mean(X[:, bs:be, :], axis=1))
        X = np.stack(X_binned, axis=1)

    # Average over trials
    X = np.mean(X, axis=2)  # now neurons x frames
    data['response'] = X

    return data


def load_processed_data(
        bin_size=1,
        smooth=False,
        smooth_win_size=5,
        downsample=False,
        downsample_n=3,
        norm_flag=True
    ):
    """Load processed data for natural movie 1.

    """

    # Set up save directories
    config = util.load_config()
    results_dir = os.path.join(config['data_dir'], 'processed', 'natural_movie_one')

    # Load data
    kwargs = {
        'bin_size': bin_size,
        'smooth': smooth,
        'smooth_win_size': smooth_win_size
    }
    data_list = util.load_results_dir(
        results_dir,
        proc_func=process_saved_movie_responses,
        kwargs=kwargs
    )

    # Convert to dataframe for ease of use and get metadata data frame
    df = pd.DataFrame(data_list)
    metadata = pd.DataFrame(list(df.metadata))
    metadata.set_index('ophys_experiment_id', inplace=True)

    # Update the dataframe structure. Because a cell can appear in multiple
    # sessions, we need to keep track of the experiment session and cell ID.
    # We also need to be able to go from experiment to stimulus type. This can
    # be done using the 'session_type' column in the metadata --
    # 'three_session_A', 'three_session_B', and 'three_session_C' should map on
    # to entries for 'natural_movie_1a', '...1b', and '...1c' in the metrics
    # CSV file.

    # Print out some statistics and get correspondence between cell ids and
    # experiment ids
    cell_ids = np.concatenate(df.neuron_ids)
    exp_ids = df.apply(lambda x: [x['exp']] * len(x['neuron_ids']), axis=1)
    exp_ids = np.concatenate(exp_ids)
    X = np.concatenate(df.response, axis=0)  # neurons x frames

    # Define cell/exp dataframe. It is a bit difficult to index based on this
    # because cells can appear in multiple experimental sessions.
    cell_exp_map = pd.DataFrame({'cell_id': cell_ids, 'exp_id': exp_ids})

    # Normalize activity if desired
    if norm_flag:
        x_mean = np.expand_dims(np.mean(X, axis=1), axis=1)
        x_std = np.expand_dims(np.std(X, axis=1), axis=1)
        X = (X - x_mean)/x_std

    return X, cell_exp_map, metadata


def plot_embedding_scatter(
        ax, Z,
        color='k',
        stimulus_info=None,
        plot_all=True,
        plot_legend=True,
        alpha=0.5
):
    """Plot t-SNE representation, color-coded by stimulus."""
    from matplotlib import patches

    # Plot all points
    if plot_all and stimulus_info is None:
        ax.scatter(Z[:, 0], Z[:, 1], s=3, c=color, alpha=alpha)

    # Plot stimulus-colored points (if specified)
    if stimulus_info is not None:
        # Get mask for invalid data points
        invalid_mask = np.logical_not(stimulus_info['mask'])
        # Plot invalid points
        if plot_all:
            ax.scatter(
                Z[invalid_mask, 0],
                Z[invalid_mask, 1],
                s=3,
                c=color,
                alpha=alpha
            )

        # Scatter plot
        ax.scatter(
            Z[stimulus_info['mask'], 0],
            Z[stimulus_info['mask'], 1],
            s=3,
            alpha=alpha,
            c=stimulus_info['color_info']['colors'][stimulus_info['mask'], :]
        )

        # Create legend object by creating proxy artists
        if plot_legend:
            legend_handles = [
                patches.Circle((0, 0), radius=3, color=v, label=k)
                for k, v in stimulus_info['color_info']['value_color_map'].items()
            ]
            ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))

    # Add axis labels
    ax.set_xlabel('Embedding dim. 1')
    ax.set_ylabel('Embedding dim. 2')

    # Plot title (if stimulus info is provided)
    if stimulus_info is not None:
        ax.set_title(f"{stimulus_info['name']}\n{stimulus_info['condition']}")

    return None


def convert_nm1_responsiveness(cell_exp_info, exp_metadata, metrics):
    """Get appropriate responsiveness values for natural movie one data.

    Because natural movie one was presented in each experimental session, it has
    three associated responsiveness values in the metrics CSV file --
    'responsiveness_nm1a', '..._nm1b', and '..._nm1c'.  Thus, if a cell was
    present for multiple sessions it can have multiple non-empty entries.
    """

    # Define a mapping from session type to responsiveness
    session_type_mapping = {
        'three_session_A': 'responsive_nm1a',
        'three_session_B': 'responsive_nm1b',
        'three_session_C': 'responsive_nm1c',
        'three_session_C2': 'responsive_nm1c'
    }
    # Define function to convert experiment type to responsiveness
    def merge_responsiveness(cei, metadata, mapping, metrics):
        col_name = mapping[metadata.loc[cei.exp_id].session_type]
        return metrics.loc[cei.cell_id][col_name]

    # Apply function to all rows of cell_exp_info
    responsiveness = cell_exp_info.apply(
        merge_responsiveness,
        axis='columns',
        args=(exp_metadata, session_type_mapping, metrics)
    )

    return responsiveness
