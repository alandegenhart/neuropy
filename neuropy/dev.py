"""Allen Institute data helper module.

This module contains general functions used when interacting with AIBS data.

"""

# Import
import os
import sys
import numpy as np
import pandas as pd
import pickle

import matplotlib
from matplotlib import cm

# Import - neuropy
import neuropy as neu

# Get home directory
home_dir = os.path.expanduser('~')


def get_criteria() -> dict:
    """Return standard analysis criteria."""

    criteria = {
        'structure': ['VISp'],
        'depth': [175],
        'cre_line': ['Slc17a7-IRES2-Cre'],
        'exp_idx': 1
    }

    return criteria


def load_data(criteria: dict):
    """Load AIBS dataset."""

    # Get valid experiment containers
    boc = load_boc()
    exp_containers = boc.get_experiment_containers(
        targeted_structures=criteria['structure'],
        imaging_depths=criteria['depth'],
        cre_lines=criteria['cre_line']
    )
    exp_containers = pd.DataFrame(exp_containers)

    # Get single experiment ID
    exps = boc.get_ophys_experiments(
        experiment_container_ids=[exp_containers.id.iloc[0]]
    )
    exps = pd.DataFrame(exps)
    exp = exps.iloc[criteria['exp_idx']]  # TODO: eventually replace with something better
    print('Experiment ID: {}, Session type: {}'.format(
        exp.id, exp.session_type
    ))

    # Load data
    exp_data = boc.get_ophys_experiment_data(ophys_experiment_id=exp.id)
    exp_events = boc.get_ophys_experiment_events(ophys_experiment_id=exp.id)

    return exp_data, exp_events


def get_design_matrix(
        data, stimulus_epochs, stimulus_list,
        remove_invalid=True
):
    """Returns a design matrix of neural responses and associated stimuli."""

    # Iterate over epochs
    X = []  # Neural responses
    Y = []  # Stimulus
    for _, e in stimulus_epochs.iterrows():
        # Get average population response during the presentation
        epoch_data = data[:, int(e.start):int(e.end)]
        epoch_data = epoch_data.mean(axis=1) * 30
        X.append(epoch_data)

        # Get stimulus information
        epoch_stim = [e[sl] for sl in stimulus_list]
        Y.append(np.array(epoch_stim))

    # Concatenate arrays
    X = np.stack(X, axis=-1)
    Y = np.stack(Y, axis=-1)

    # Remove invalid entries
    if remove_invalid:
        valid_mask = np.logical_and(np.isfinite(Y[0, :]), np.isfinite(Y[1, :]))
        X = X[:, valid_mask]
        Y = Y[:, valid_mask]

    return X, Y


def calculate_tuning(X, Y):
    """Calculate mean response/tuning."""
    import itertools

    # Get stimulus sets -- these are the unique values for each row of Y
    n_stimuli = Y.shape[0]
    Y_unique = []
    for i in range(n_stimuli):
        y = Y[i, :]
        y_unique = np.unique(y)
        y_sorted = np.sort(y_unique)
        Y_unique.append(y_sorted)

    # Calculate mean response for stimulus sets
    stim_val_comb = itertools.product(*Y_unique)
    array_sz = [X.shape[0]]
    array_sz.extend([len(yu) for yu in Y_unique])
    X_avg = np.full(array_sz, np.nan)
    for sv in stim_val_comb:
        # Find all valid observations for the stimulus set
        mask = []
        si = []  # Stimulus inds
        for i, s in enumerate(sv):
            # Get all instances in Y of the current stimulus
            m = Y[i, :] == s
            si.append(np.argwhere(Y_unique[i] == s)[0, 0])
            mask.append(m)

        # Find overall mask and get neural response
        mask = np.logical_and(*mask)
        X_stim = X[:, mask]
        X_stim = np.mean(X_stim, axis=1)

        # Add response to X_avg.  This requires supplying a tuple as the index,
        # which allows an arbitrary number of dimensions to be used.
        resp_idx = [Ellipsis]
        resp_idx.extend(si)
        X_avg[tuple(resp_idx)] = X_stim

    return X_avg, Y_unique


def plot_tuning(X, Y_stimuli, md, save_dir):
    """Plot tuning for all neurons/states.

    NOTE: currently this is assuming a standard structure of data, where the
    stimuli are orientation and temporal frequency, respectively.
    """
    from matplotlib import cm

    # Get number of orientation and temporal frequency stimuli
    orientation = Y_stimuli[0]
    temporal_frequency = Y_stimuli[1]
    n_orientation = orientation.shape[0]
    n_temporal_frequency = temporal_frequency.shape[0]

    # Create color maps.  For orientation tuning use a circular color map
    orientation_colormap = cm.ScalarMappable(cmap='hsv')
    orientation_colors = orientation_colormap.to_rgba(range(n_orientation + 1))  # Need + 1 to wrap correctly
    temporal_frequency_colormap = cm.ScalarMappable(cmap='plasma')
    temporal_frequency_colors = temporal_frequency_colormap.to_rgba(range(n_temporal_frequency))

    # Iterate over neurons/states
    n_neurons = X.shape[0]
    for n in range(n_neurons):
        # Create figure
        fh, axh = neu.temp.subplot_fixed(
            1, 2, [600, 300],
            x_margin=[200, 200],
            y_margin=[200, 300])

        # Plot orientation tuning for each temporal frequency
        curr_ax = axh[0][0]
        for i, f in enumerate(temporal_frequency):
            x = X[n, :, i]
            curr_ax.plot(
                orientation, x, 'o-',
                label=str(f),
                color=temporal_frequency_colors[i, :]
            )

        curr_ax.set_xticks(orientation)
        curr_ax.set_xlabel('Orientation')
        curr_ax.set_ylabel('Response')
        curr_ax.legend()

        # Plot temporal frequency tuning for each direction
        curr_ax = axh[0][1]
        for i, o in enumerate(orientation):
            x = X[n, i, :]
            curr_ax.plot(
                temporal_frequency, x, 'o-',
                label=str(o),
                color=orientation_colors[i, :]
            )

        curr_ax.set_xticks(temporal_frequency)
        curr_ax.set_xlabel('Temporal frequency')
        curr_ax.set_ylabel('Response')
        curr_ax.legend()

        # Add title string
        info = [f"PC: {n}"]
        aibs.add_figure_title_info(fh, md, additional_info=info)

        # Save figure
        fig_name_str = [
            'Exp', str(md['ophys_experiment_id']), 'PCTuning', 'PC', str(n)
        ]
        fig_name_str = '_'.join(fig_name_str) + '.pdf'
        fig_save_path = os.path.join(save_dir, fig_name_str)
        fh.savefig(fig_save_path)

    return None


def smooth_responses(X, win_size=15, ds=True, ds_n=30):
    """Smooth and downsample responses."""

    # Define window to smooth responses
    win = np.hanning(win_size)

    # Iterate over neurons and smooth
    n_neurons = X.shape[0]
    n_frames = X.shape[1]
    n_trials = X.shape[2]
    X_sm = np.full(X.shape, np.nan)  # Neuron x frames x trials
    for nrn in range(n_neurons):
        for trl in range(n_trials):
            x = X[nrn, :, trl]
            x = np.convolve(x, win, mode='same')
            X_sm[nrn, :, trl] = x

    # Downsample if desired
    if ds:
        onset = np.floor(ds_n/2).astype(int)
        ds_idx = np.arange(onset, n_frames, ds_n)
        X_sm = X_sm[:, ds_idx, :]

    return X_sm


def cut_drifting_grating_data(exp_data, exp_events):
    """Get design matrix of averaged data for drifting gratings."""

    # Get stimulus information, epochs, etc.
    stimulus_epochs = exp_data.get_stimulus_table('drifting_gratings')

    return X, stimulus_info


def cut_movie_data(exp_data, exp_events):
    """Cut movie data into a design matrix."""
    import itertools

    # Get movie information, including number of frames and repeats
    stimulus_epochs = exp_data.get_stimulus_table('natural_movie_one')
    frame_num = np.sort(stimulus_epochs.frame.unique())
    repeats = np.sort(stimulus_epochs.repeat.unique())

    # Initialize design matrix
    n_neurons = exp_events.shape[0]
    n_frames = frame_num.shape[0]
    n_repeats = repeats.shape[0]
    X = np.full((n_neurons, n_frames, n_repeats), np.nan)

    # Iterate over frame/repeat combinations.  While this is inefficient, it is
    # necessary because there can be an offset in the time indices corresponding
    # to each frame.  The difference between the start and end indices can be
    # {0, 1, 2}.  This is likely due to frame rate differences (?).  For now,
    # just take one sample corresponding to the 'start' index.
    for r in repeats:
        # Get epochs for the current repeat
        repeat_epochs = stimulus_epochs[stimulus_epochs.repeat == r]

        # It seems as if the frames are always organized in increasing order.
        # However, perform a few checks here just to confirm.

        # Check to make sure the correct number of frames exist
        n_rows = repeat_epochs.shape[0]
        if n_rows != frame_num.shape[0]:
            print(f'Warning: invalid number of frames found for repeat {r}.')

        # Check to make sure the frames increase in ascending order
        unique_diff = np.unique(np.diff(repeat_epochs.frame))
        if unique_diff.shape[0] != 1:
            print(f'Warning: frames are not ordered for repeat {r}.')

        # Check to make sure the unique number of start indices equals the
        # number of frames
        unique_start_inds = repeat_epochs.start.unique()
        if unique_start_inds.shape[0] != n_frames:
            print(f'Warning: number of unique start indices is less than the number of frames for repeat {r}.')

        # If all the checks pass, use the start indices to get the data
        repeat_inds = repeat_epochs.start
        X[:, :, r] = exp_events[:, repeat_inds]

    return X, frame_num, repeats

