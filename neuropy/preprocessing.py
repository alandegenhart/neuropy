"""Preprocessing module

This module contains code used to perform various preprocessing actions, such
as cutting data for specific stimuli.
"""

# Import
import os
import sys
import numpy as np
import pandas as pd
import pickle

import matplotlib
from matplotlib import cm

# Import - custom modules
import neuropy
import neuropy.movie
import neuropy.dev
import neuropy.plot

# Get home directory
home_dir = os.path.expanduser('~')


def process_movie_data(exp_data, exp_events, metadata, save_dir):

    # Cut movie data into neurons x frames x repeats
    X, frame_num, repeats = cut_movie_data(exp_data, exp_events)

    # Plot single-trial responses
    fh = movie.plot_single_trial_movie_response(X)
    neuropy.plot.add_figure_title_info(fh, metadata)
    fig_name = f"Exp_{metadata['ophys_experiment_id']}_MovieResponse_SingleTrial.pdf"
    fh.savefig(os.path.join(save_dir, fig_name))

    # Plot trial-averaged responses
    fh = movie.plot_trial_averaged_movie_response(np.mean(X, axis=2), frame_num)
    neuropy.plot.add_figure_title_info(fh, metadata)
    fig_name = f"Exp_{metadata['ophys_experiment_id']}_MovieResponse_TrialAveraged.pdf"
    fh.savefig(os.path.join(save_dir, fig_name))

    # Create dict with stimulus info
    stimulus_info = {
        'frames': frame_num,
        'repeats': repeats
    }

    return X, stimulus_info


def cut_movie_data(exp_data, exp_events, bin_size=1, movie_name='natural_movie_one'):
    """Cut movie data into a design matrix."""
    import itertools

    # Get movie information, including number of frames and repeats
    stimulus_epochs = exp_data.get_stimulus_table(movie_name)
    frame_num = np.sort(stimulus_epochs.frame.unique())
    repeats = np.sort(stimulus_epochs.repeat.unique())

    # Initialize design matrix
    n_neurons = exp_events.shape[0]
    n_frames = frame_num.shape[0]
    n_bins = int(np.ceil(n_frames/bin_size))
    n_repeats = repeats.shape[0]
    X = np.full((n_neurons, n_bins, n_repeats), np.nan)

    # Iterate over frame/repeat combinations.  While this is inefficient, it is
    # necessary because there can be an offset in the time indices corresponding
    # to each frame.  The difference between the start and end indices can be
    # {0, 1, 2}.  This is likely due to frame rate differences (?).  For now,
    # just take one sample corresponding to the 'start' index.
    for r in repeats:
        # Get epochs for the current repeat
        repeat_epochs = stimulus_epochs[stimulus_epochs.repeat == r]

        # Get n_frames of data starting at the timestamp corresponding to frame
        # 0. This is not guaranteed to perfectly match up with the actual
        # stimulus being presented, *but* should avoid any discontinuities.
        # Furthermore, any discrepancies are likely to only be a few (<5) time
        # steps (approx. 150ms).
        start_ind = repeat_epochs.start[repeat_epochs.frame == 0].iloc[0]
        X_repeat = exp_events[:, start_ind:(start_ind + n_frames)]
        X_repeat = bin_data(X_repeat, bin_size=bin_size)
        X[:, :, r] = X_repeat

    return X, frame_num, repeats


def process_drifting_gratings_data(exp_data, exp_events, metadata, save_dir):
    """Get processed data during drifting gratings stimulus presentation."""

    # Generate design matrix
    X, stimulus_info = cut_drifting_gratings_data(exp_data, exp_events)

    # Create plots (if desired)

    return X, stimulus_info


def cut_drifting_gratings_data(exp_data, exp_events, bin_size=1):
    """Cut neural responses during presentation of drifting gratings and
    return stimulus table with cut and binned data for each presentation.
    """

    # Get stimulus epochs, rename orientation as direction, and convert
    # orientation to direction
    stimulus_table = exp_data.get_stimulus_table('drifting_gratings')
    stimulus_table.drop(columns='blank_sweep', inplace=True)
    stimulus_table.dropna(subset=['temporal_frequency', 'orientation'], inplace=True)
    stimulus_table['direction'] = stimulus_table.orientation
    stimulus_table.orientation = stimulus_table.orientation.map(lambda x: x - 180 if x >= 180.0 else x)

    # Bin dff and events
    _, dff = exp_data.get_dff_traces()
    binned_dff = bin_by_stimulus_table(stimulus_table, dff, bin_size=bin_size)
    binned_events = bin_by_stimulus_table(stimulus_table, exp_events, bin_size=bin_size)

    # Add binned data to DataFrame
    stimulus_table['dff'] = binned_dff
    stimulus_table['events'] = binned_events

    return stimulus_table


def get_proc_function(stimulus):
    """Get function used to preprocess data for a specified stimulus."""

    # Define mapping from stimuli to processing functions
    proc_func_dict = {
        'natural_movie_one': process_movie_data,
        'drifting_gratings': process_drifting_gratings_data
    }

    return proc_func_dict[stimulus]


def bin_by_stimulus_table(stimulus_table, X, bin_size=1):
    """Bin data using stimulus start/stop indices."""
    # Define wrapper around binning function
    def bin_func(row, X, bin_size):
        return neuropy.preprocessing.bin_data(X[:, row.start.astype(int):row.end.astype(int)], bin_size=bin_size)
    # Bin dff and events
    return stimulus_table.apply(bin_func, axis=1, args=(X, bin_size))


def bin_data(X, bin_size=1, bin_func=np.sum, incomplete=False):
    """Bin data across samples.

    Inputs:
        X   Data to bin (n_features, n_samples)
        bin_size   Number of samples to include in each bin
        bin_func   Function used for binning (e.g., np.sum, np.mean)

    Outputs
        X_binned   Binned data (n_features, n_bins)
    """

    # If bin size is 1, return the original data. 
    if bin_size == 1:
        return X

    # Get start and end of bins. Note that the last bin will be truncated
    # In the future, this could be updated to allow the user to specify
    # how bins at the end of the data are handled (either truncated or discarded)
    bin_start = np.arange(0, X.shape[1], bin_size)
    bin_end = bin_start + bin_size

    # Determine the indices of the final bin. If 'incomplete' is TRUE, then
    # the final bin can be partial. If 'incomplete' is FALSE, then only the
    # indices corresponding to full bins are used.
    #
    # Note that if incomplete is set to TRUE, the bin function should be np.mean.
    # Otherwise, the number of events in the final bin will be lower than
    # expected.
    if incomplete:
        bin_end[-1] = X.shape[1]
    elif bin_end[-1] != X.shape[1]:
        # Discard the last start/end index
        bin_start = bin_start[0:-1]
        bin_end = bin_end[0:-1]
    else:
        # Do nothing -- number of elements in X is a multiple of the bin size
        pass

    # Iterate over indices and bin data
    X_binned = []
    for bs, be in zip(bin_start, bin_end):
        X_binned.append(bin_func(X[:, bs:be], axis=1))
    X_binned = np.stack(X_binned, axis=1)

    return X_binned

