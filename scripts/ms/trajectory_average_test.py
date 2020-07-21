"""Figure 4 main analysis

"""


# Import libraries
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import multiprocessing as mp
import argparse

# Define directory paths and import modules
home_dir = os.path.expanduser('~')
src_dir = os.path.join(home_dir, 'src', 'neuropy')
sys.path.append(src_dir)
import neuropy as neu
from neuropy.analysis import gpfa
import neuropy.temp as tmp
import neuropy.el.ms.fig_4 as f4


def main():
    # Parameters
    location = 'ssd'
    subject = 'Earl'
    dataset = 20190307  # Only two-target intuitive this day

    # Get valid datasets
    EL = neu.el.util.ExperimentLog()
    EL.load()
    EL.get_data_path(location)
    criteria = neu.el.util.get_valid_criteria()
    criteria['subject'] = subject
    criteria['dataset'] = dataset
    EL.apply_criteria(criteria)

    # Get experiment sets
    task_list = ['tt_int']
    experiment_list = EL.get_experiment_sets(task_list)
    # Define data and results locations
    data_path, results_path = neu.el.util.get_data_path(location)

    # Load data
    data_dir, _ = os.path.split(experiment_list.iloc[0]['dir_path'])
    int_file = experiment_list.iloc[0]['tt_int'][0][0] + '_pandasData.hdf'

    """Step 1: Load data, decoding, and GPFA parameters."""
    # Load data
    pandas_dir = os.path.join(data_dir, 'pandasData')
    df_int = pd.read_hdf(os.path.join(pandas_dir, int_file), 'df')

    # Drop unused columns to save space
    drop_cols = [
        'acc', 'intTargPos', 'intTargSz', 'spikes', 'tag', 'trialName', 'tube',
        'vel', 'pos'
    ]
    df_int.drop(columns=drop_cols, inplace=True)

    # Get decoder names
    int_dec_path = os.path.join(
        data_dir, df_int['decoderName'].iloc[0] + '.mat')

    # Load decoding parameters and GPFA results
    dec_int = neu.util.convertmat.convert_mat(int_dec_path)['bci_params']
    neu.el.proc.clean_bci_params(dec_int)

    # Define paths to GPFA data
    # TODO: update this to use actual decoder number?
    gpfa_results_dir = os.path.join(data_dir, 'analysis', 'mat_results')
    int_dec_num = 5  # Need to re-convert data to get this from the params
    int_gpfa_path = os.path.join(
        gpfa_results_dir,
        'run{:03d}'.format(int_dec_num),
        'gpfa_xDim{}.mat'.format(dec_int['xDim'])
    )
    int_gpfa_path = os.path.join(gpfa_results_dir, int_gpfa_path)

    # Load GPFA data
    gpfa_int = neu.util.convertmat.convert_mat(int_gpfa_path)['estParams']
    neu.el.proc.clean_gpfa_params(gpfa_int)

    # Find onset/offset and add back to dataframe.
    onset_idx_int = df_int.apply(tmp.find_traj_onset,
                                 axis=1, result_type='expand')
    df_int['trajOnsetIdx'] = onset_idx_int['trajOnset']
    df_int['trajOffsetIdx'] = onset_idx_int['trajOffset']

    # Remove non-paired targets
    df_int, targ_cond_int, targ_info_int = tmp.remove_non_paired_trials(df_int)

    """Step 2: Extract neural trajectories."""

    # Orthonormalize
    C_orth, T, s, VH = gpfa.orthogonalize(gpfa_int['C'])
    x_dim = T.shape[0]

    # Get neural trajectories for intuitive mapping
    U_int = df_int['decodeSpikeCounts'].apply(
        neu.analysis.gpfa.extract_causal_traj, args=(dec_int,)
    )

    # Limit trajectories to the valid portion for each trial. Might be possible
    # to do this using list comprehension, but it is a bit tricky here b/c the
    # neural trajectories have been removed from the dataframe

    # Truncate intuitive trajectories
    for i in range(U_int.shape[0]):
        idx = range(df_int['trajOnsetIdx'].iloc[i],
                    df_int['trajOffsetIdx'].iloc[i] + 1)
        U_int.iloc[i] = U_int.iloc[i][:, idx]

    # Transform to orthonormalized latents
    # NOTE -- the rotated trials have been removed
    U = {'int': U_int}
    U_orth = {dec: U_dec.apply(lambda u: T @ u) for dec, U_dec in U.items()}

    # Get unique targets to plot. Should only have to do this for the intuitive
    # trials b/c both the intuitive and rotated use the same target config.
    targ_cond_unique = set(targ_cond_int)

    """Step 3: Average trajectories"""

    # TODO: consider updating onset/offset indices to include the first step
    #   of the target state

    # Truncate trajectories and get series with the decode state
    def truncate_traj(row):
        onset_idx = row['trajOnsetIdx']
        offset_idx = row['trajOffsetIdx'] + 1
        row['decodeState'] = row['decodeState'][:, onset_idx:offset_idx]
        return row
    df_int = df_int.apply(truncate_traj, axis=1)
    traj = df_int['decodeState']
    targ_cond_int = pd.Series(targ_cond_int)

    # Setup axes
    fh, axh = tmp.subplot_fixed(1, len(targ_cond_unique), [300, 300])
    col_map = tmp.define_color_map()

    # Iterate over target pairs and plot
    for idx, tcu in enumerate(targ_cond_unique):
        targ_mask = [True if tc == tcu else False for tc in targ_cond_int]
        # Plot trajectories
        curr_ax = axh[0][idx]
        tmp.plot_traj(
            traj[targ_mask],
            targ_cond_int[targ_mask],
            col_map,
            axh=curr_ax,
            col_mode='light'
        )

        # Get data and fit model
        traj_trunc, traj_diff = tmp.get_traj_velocity(traj[targ_mask])
        traj_trunc = np.concatenate(traj_trunc.to_numpy(), axis=1)  # dim x samp
        traj_diff = np.concatenate(traj_diff.to_numpy(), axis=1)  # dim x samp
        model = fit_trajectory_average(traj_trunc.T, traj_diff.T)

        # Get start and end of trajectories
        initial_pos = traj[targ_mask].apply(lambda x: x[:, 0:1])
        initial_pos = np.concatenate(initial_pos.to_numpy(), axis=1)
        initial_pos_mean = np.mean(initial_pos, axis=1, keepdims=True)
        end_pos = traj[targ_mask].apply(lambda x: x[:, -1:])
        end_pos = np.concatenate(end_pos.to_numpy(), axis=1)
        end_pos_mean = np.mean(end_pos, axis=1, keepdims=True)
        end_pos_std = np.std(end_pos, axis=1, keepdims=True)
        d_thresh = end_pos_std.mean()

        # Iterate over time steps
        converged = False
        current_pos = initial_pos_mean
        predicted_pos = [current_pos]
        max_iter = 100
        scale_factor = 0.5
        i = 0
        while not converged:
            i += 1

            # Predict velocity and add to current position
            v = model.predict(current_pos.T)
            current_pos = current_pos + v.T * scale_factor
            predicted_pos.append(current_pos)

            # Calculate the distance from the endpoint pos
            d = np.linalg.norm(current_pos - end_pos_mean)
            if d < d_thresh:
                converged = True

            # Exit if maximum iterations has been hit
            if i > max_iter:
                print('Warning: trajectory did not converge.')
                break

        # Plot predicted position
        predicted_pos = pd.Series([np.concatenate(predicted_pos, axis=1)])
        tmp.plot_traj(
            predicted_pos,
            pd.Series([tcu]),
            col_map,
            axh=curr_ax,
            col_mode='dark',
            line_width=2
        )

        # Format axes
        curr_ax.set_xlim([-150, 150])
        curr_ax.set_ylim([-150, 150])

    # Save figure
    save_dir = os.path.join(results_path, 'el_ms', 'average_test')
    fig_str = '{}_{}_decodedTraj.pdf'.format(subject, dataset)
    fh.savefig(os.path.join(save_dir, fig_str))

    return None


def fit_trajectory_average(pos, vel):
    """Fit trajectory average."""
    units = [20, 20, 10]
    epochs = 500
    act_fn = tf.keras.activations.relu
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(units[0], activation=act_fn, input_shape=[2]),
            tf.keras.layers.Dense(units[1], activation=act_fn),
            tf.keras.layers.Dense(units[2], activation=act_fn),
            tf.keras.layers.Dense(2, activation=tf.keras.activations.linear)
        ]
    )
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.mean_squared_error
    )
    model.summary()
    # Train model
    model.fit(pos, vel, epochs=epochs)

    return model


# Call main function
if __name__ == '__main__':
    main()
