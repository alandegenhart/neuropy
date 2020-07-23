"""Figure 4 main analysis

"""


# Import libraries
import os
import sys
import pandas as pd
import numpy as np
import scipy as sp
import tensorflow as tf

# Define directory paths and import modules
home_dir = os.path.expanduser('~')
src_dir = os.path.join(home_dir, 'src', 'neuropy')
sys.path.append(src_dir)
import neuropy as neu
from neuropy.analysis import gpfa
import neuropy.temp as tmp


def main():
    # --- Define dataset parameters and load data ---
    # Dataset parameters
    ds_info = {
        'location': 'ssd',
        'subject': 'Earl',
        'dataset': 20180927,
        'criteria_set': 'two_target_int',
        'dec_num': 5,
        'targ_pair_idx': 0
    }
    save_dir = os.path.join(
        os.path.expanduser('~'), 'results', 'el_ms', 'proj_test'
    )
    os.makedirs(save_dir, exist_ok=True)

    # NOTE -- For the 20180927 dataset, the GPFA rot criteria is different
    # this is currently specified in the 'load_data' function, but should be
    # moved out of here eventually.

    # Load data
    df, dec, gpfa_params, targ_cond, targ_info = load_data(ds_info)

    # --- Plot cursor trajectories ---

    # Truncate trajectories and get series with the decode state
    df = df.apply(truncate_traj, axis=1)
    # Plot
    col_map = tmp.define_color_map()
    fig_title_str = '{} {} :: Cursor trajectories :: Target pair {}'.format(
        ds_info['subject'], ds_info['dataset'], ds_info['targ_pair_idx']
    )
    fh = create_cursor_traj_plot(
        df['decodeState'], pd.Series(targ_cond), col_map, fig_title_str
    )
    fig_name_str = '{}_{}_0_cursorTrajectories_targPair_{}.pdf'.format(
        ds_info['subject'], ds_info['dataset'], ds_info['targ_pair_idx']
    )
    fh.savefig(os.path.join(save_dir, fig_name_str))

    # --- Extract neural trajectories ---
    # Orthonormalize
    C_orth, TT, s, VH = gpfa.orthogonalize(gpfa_params['C'])

    # Get neural trajectories for intuitive mapping
    U = df['decodeSpikeCounts'].apply(
        neu.analysis.gpfa.extract_causal_traj, args=(dec,)
    )

    # Limit trajectories to the valid portion for each trial. Might be possible
    # to do this using list comprehension, but it is a bit tricky here b/c the
    # neural trajectories have been removed from the dataframe
    for i in range(U.shape[0]):
        idx = range(df['trajOnsetIdx'].iloc[i],
                    df['trajOffsetIdx'].iloc[i] + 1)
        U.iloc[i] = U.iloc[i][:, idx]

    # Transform to orthonormalized latents
    U_orth = U.apply(lambda u: TT @ u)

    # Get features used in identifying projections
    asym_features = neu.el.comp.get_asymmetry_features(U_orth, targ_cond)

    # --- Calculate orthonormal basis of intuitive mapping and null space ---
    # The decoder weights map non-orthonormalized latent activity to cursor
    # position.  Since we want to work with orthonormalized latents, we need to
    # transform the weights.
    #
    # The weights transform the latents to decoder position:
    # p = W @ u
    #
    # We want a new set of weights W_orth that produce:
    # p = W_orth @ u_orth
    #
    # Using the transformation matrix T, which satisfies:
    # u_orth = TT @ u
    #
    # The set of weights mapping the orthonormalized latents to cursor position
    # are:
    # W_orth = W @ TT' @ (TT @ TT')^-1
    W_orth = dec['W'] @ TT.T @ np.linalg.inv(TT @ TT.T)
    W_orthnorm, _, _, _ = gpfa.orthogonalize(W_orth.T)
    W_null = sp.linalg.null_space(W_orthnorm.T)  # W_null: 10 x 8

    # Project neural activity into null and potent spaces
    U_null = U_orth.apply(lambda u: W_null.T @ u)
    U_potent = U_orth.apply(lambda u: W_orthnorm.T @ u)

    # Run LDA
    X_center_null = W_null.T @ asym_features['x_center']
    D_lda, J = neu.analysis.math.fisher_lda(
        X_center_null, asym_features['cond'])
    W_lda = W_null @ D_lda
    
    # Method 2 -- optimization-based
    #p = fit_asymmetry_axis(asym_features, W_null)
    #W_lda = W_null @ p

    # Define new set of projection vectors consisting of the intuitive space
    # and 1st LDA dimension
    W_combined = np.concatenate([W_orthnorm, W_lda], axis=1)
    U_combined = U_orth.apply(lambda u: W_combined.T @ u)

    # --- Plot 2D projections ---
    # Define plotting parameters
    ax_lim = [-10, 10]
    ax_tick = [ax_lim[0], 0, ax_lim[1]]

    # Plot 2D projections
    title_str = '{} {} :: Projections :: Target pair {}'.format(
        ds_info['subject'], ds_info['dataset'], ds_info['targ_pair_idx']
    )
    fh_2d = create_2d_plot(
        U_combined, pd.Series(targ_cond), ax_lim, ax_tick, col_map, title_str
    )

    # Save 2D plot
    fig_name_str = '{}_{}_1_projections_targPair_{}.pdf'.format(
        ds_info['subject'], ds_info['dataset'], ds_info['targ_pair_idx']
    )
    fig_name = os.path.join(save_dir, fig_name_str)
    fh_2d.savefig(fig_name)

    # --- 3D plot of trajectories ---
    # Create plot title
    fig_title_str = '{} {} :: 3D projection :: Target pair {}'.format(
        ds_info['subject'], ds_info['dataset'], ds_info['targ_pair_idx']
    )
    # Plot
    fh_3d = create_3d_plot(
        U_combined, pd.Series(targ_cond), col_map, ax_lim, ax_tick,
        fig_title_str,
        col_mode='light',
        line_width=1
    )
    # Save figure
    fig_name_str = '{}_{}_2_3dProj_targPair_{}.pdf'.format(
        ds_info['subject'], ds_info['dataset'], ds_info['targ_pair_idx']
    )
    fig_name = os.path.join(save_dir, fig_name_str)
    fh_3d.savefig(fig_name)

    # --- Fit trajectory averages and plot ---
    # Average trajectories
    avg_traj, avg_cond = average_trajectories(U_combined, targ_cond)
    # Plot average trajectories
    fig_title_str = '{} {} :: 3D projection (averaged) :: Target pair {}'.format(
        ds_info['subject'], ds_info['dataset'], ds_info['targ_pair_idx']
    )
    fh_3d = create_3d_plot(
        avg_traj, pd.Series(avg_cond), col_map, ax_lim, ax_tick, fig_title_str,
        col_mode='dark',
        line_width=3
    )
    # Save figure
    fig_name_str = '{}_{}_3_3dProj_avg_targPair_{}.pdf'.format(
        ds_info['subject'], ds_info['dataset'], ds_info['targ_pair_idx']
    )
    fig_name = os.path.join(save_dir, fig_name_str)
    fh_3d.savefig(fig_name)
    
    return None


def load_data(ds_info):
    """Load pandas data for a single dataset."""

    # Get valid datasets
    EL = neu.el.util.ExperimentLog()
    EL.load()
    EL.get_data_path(ds_info['location'])
    criteria = neu.el.util.get_valid_criteria(ds_info['criteria_set'])
    criteria['subject'] = ds_info['subject']
    criteria['dataset'] = ds_info['dataset']
    criteria['gpfa_rot'] = 'cond_4'
    criteria['align'] = 'start, possible bug'
    EL.apply_criteria(criteria)

    # Get experiment sets
    task_list = ['tt_int']
    experiment_list = EL.get_experiment_sets(task_list)
    # Define data and results locations
    data_path, results_path = neu.el.util.get_data_path(ds_info['location'])

    # TODO: Load data for all target pairs here

    # Load data
    targ_pair_idx = 0
    data_dir, _ = os.path.split(
        experiment_list.iloc[ds_info['targ_pair_idx']]['dir_path']
    )
    pandas_file = experiment_list.iloc[ds_info['targ_pair_idx']]['tt_int'][0][0] \
        + '_pandasData.hdf'

    """Step 1: Load data, decoding, and GPFA parameters."""
    # Load data
    pandas_dir = os.path.join(data_dir, 'pandasData')
    df = pd.read_hdf(os.path.join(pandas_dir, pandas_file), 'df')

    # Drop unused columns to save space
    drop_cols = [
        'acc', 'intTargPos', 'intTargSz', 'spikes', 'tag', 'trialName', 'tube',
        'vel', 'pos'
    ]
    df.drop(columns=drop_cols, inplace=True)

    # Get decoder names
    dec_path = os.path.join(
        data_dir, df['decoderName'].iloc[0] + '.mat')

    # Load decoding parameters and GPFA results
    dec = neu.util.convertmat.convert_mat(dec_path)['bci_params']
    neu.el.proc.clean_bci_params(dec)

    # Define paths to GPFA data
    gpfa_results_dir = os.path.join(data_dir, 'analysis', 'mat_results')
    gpfa_path = os.path.join(
        gpfa_results_dir,
        'run{:03d}'.format(ds_info['dec_num']),
        'gpfa_xDim{}.mat'.format(dec['xDim'])
    )
    gpfa_path = os.path.join(gpfa_results_dir, gpfa_path)

    # Load GPFA data
    gpfa_params = neu.util.convertmat.convert_mat(gpfa_path)['estParams']
    neu.el.proc.clean_gpfa_params(gpfa_params)

    # Find onset/offset and add back to dataframe.
    onset_idx_int = df.apply(tmp.find_traj_onset, axis=1, result_type='expand')
    df['trajOnsetIdx'] = onset_idx_int['trajOnset']
    df['trajOffsetIdx'] = onset_idx_int['trajOffset']

    # Remove failed and non-paired targets
    df = df[df['successful']]
    df, targ_cond, targ_info = tmp.remove_non_paired_trials(df)

    return df, dec, gpfa_params, targ_cond, targ_info


def truncate_traj(row):
    """Truncate trajectories for a single trial."""
    onset_idx = row['trajOnsetIdx']
    offset_idx = row['trajOffsetIdx'] + 1
    row['decodeState'] = row['decodeState'][:, onset_idx:offset_idx]
    return row


def fit_asymmetry_axis(asym_features, W):
    """Find an asymmetry axis using gradient descent optimization.

    Note: this approach does not look for an axis with unit length. A l2
    penalty is used in the cost function in order to discourage the length of
    the axis from getting too large.  However, a more principled approach such
    as optimization over the Stiefel manifold is probably more appropriate.

    """
    M = W.shape[1]  # Number of dimensions
    
    # Create variables and constants
    init_val = np.random.rand(M, 1)
    init_val = init_val / np.linalg.norm(init_val)
    p = tf.Variable(initial_value=init_val)
    mu_ab = tf.constant(W.T @ asym_features['mu_center'][0])
    mu_ba = tf.constant(W.T @ asym_features['mu_center'][1])
    mu_a = tf.constant(W.T @ asym_features['mu_start'][0])
    mu_b = tf.constant(W.T @ asym_features['mu_start'][1])

    # Define loss
    def loss():
        d_start = (tf.transpose(p) @ (mu_a - mu_b)) ** 2
        d_mid = (tf.transpose(p) @ (mu_ab - mu_ba)) ** 2
        l2 = tf.transpose(p) @ p
        return d_start - d_mid + l2

    # Create optimizer object and minimize
    opt = tf.keras.optimizers.SGD(learning_rate=0.0001)
    epochs = 100
    J = []
    for _ in range(epochs):
        opt.minimize(loss, var_list=[p])
        J.append(loss().numpy())
    
    # Normalize vector
    p_final = p.numpy()
    p_final = p_final / np.linalg.norm(p_final)

    return p_final
    

def fit_trajectory_model(pos, vel):
    """Fit trajectory average."""
    input_dim = pos.shape[1]
    units = [20, 20, 10]
    epochs = 500
    act_fn = tf.keras.activations.relu
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(units[0], activation=act_fn, input_shape=[input_dim]),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units[1], activation=act_fn),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units[2], activation=act_fn),
            tf.keras.layers.Dense(input_dim, activation=tf.keras.activations.linear)
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


def average_trajectories(traj, targ_cond):
    """Average trajectories using neural network model."""

    # Get unique target conditions
    targ_cond_unique = list(set(targ_cond))

    # Iterate over target pairs and plot
    avg_traj = []
    for idx, tcu in enumerate(targ_cond_unique):
        targ_mask = [True if tc == tcu else False for tc in targ_cond]

        # Get data and fit model
        traj_trunc, traj_diff = tmp.get_traj_velocity(traj[targ_mask])
        traj_trunc = np.concatenate(traj_trunc.to_numpy(), axis=1)  # dim x samp
        traj_diff = np.concatenate(traj_diff.to_numpy(), axis=1)  # dim x samp
        model = fit_trajectory_model(traj_trunc.T, traj_diff.T)

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
        scale_factor = 1
        threshold_check = False
        i = 0
        while not converged:
            i += 1

            # Predict velocity and add to current position
            v = model.predict(current_pos.T)
            current_pos = current_pos + v.T * scale_factor
            predicted_pos.append(current_pos)

            # Calculate the distance from the endpoint pos
            d = np.linalg.norm(current_pos - end_pos_mean)
            if (d < d_thresh) and threshold_check:
                predicted_pos = np.concatenate(predicted_pos, axis=1)
                converged = True

            # Exit if maximum iterations has been hit
            if i > max_iter:
                print('Warning: trajectory did not converge and has been truncated.')
                # Truncate trajectory
                predicted_pos = np.concatenate(predicted_pos, axis=1)
                d = np.linalg.norm(predicted_pos - end_pos_mean, axis=0)
                d_idx = np.argmin(d)
                predicted_pos = predicted_pos[:, 0:(d_idx+1)]

                break

        # Plot predicted position
        avg_traj.append(predicted_pos)

    # Convert to series
    avg_traj = pd.Series(avg_traj)

    return avg_traj, targ_cond_unique


def create_cursor_traj_plot(pos, cond, col_map, title_str):
    """Plot cursor trajectories.
    """
    # Setup figure and axis
    fh, axh = tmp.subplot_fixed(
        1, 1, [300, 300],
        x_margin=[150, 150],
        y_margin=[150, 200]
    )
    # Plot
    curr_ax = axh[0][0]
    tmp.plot_traj(
        pos,
        cond,
        col_map,
        axh=curr_ax,
        col_mode='light'
    )

    # Format plot
    ax_lim = [-150, 150]
    ax_tick = [ax_lim[0], 0, ax_lim[1]]
    curr_ax.set_xlim(ax_lim)
    curr_ax.set_ylim(ax_lim)
    curr_ax.set_xticks(ax_tick)
    curr_ax.set_yticks(ax_tick)
    curr_ax.set_xlabel('Cursor pos. (x)')
    curr_ax.set_ylabel('Cursor pos. (y)')

    # Set title
    fh.suptitle(title_str)

    return fh


def create_2d_plot(traj, cond, ax_lim, ax_tick, col_map, title_str):
    """Create 2d projection plots."""

    # Setup axes and axis limits
    dims = [[0, 1], [0, 2], [1, 2]]  # Sets of dimensions to plot
    fh, axh = tmp.subplot_fixed(
        1, len(dims), [300, 300],
        x_margin=[150, 150],
        y_margin=[150, 200]
    )

    # Iterate over projections and plot
    for plot_no, d in enumerate(dims):
        # Get series for selected dimensions
        traj_d = traj.apply(lambda u: u[d, :])

        # Plot
        curr_ax = axh[0][plot_no]
        tmp.plot_traj(
            traj_d,
            cond,
            col_map,
            axh=curr_ax,
            col_mode='light'
        )
        curr_ax.set_xlim(ax_lim)
        curr_ax.set_ylim(ax_lim)
        curr_ax.set_xticks(ax_tick)
        curr_ax.set_yticks(ax_tick)
        curr_ax.set_xlabel('Dim. {}'.format(d[0] + 1))
        curr_ax.set_ylabel('Dim. {}'.format(d[1] + 1))

    # Set figure title
    fh.suptitle(title_str)

    return fh


def create_3d_axes():
    """Create 3d axis in new figure."""

    import matplotlib as mpl
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


def create_3d_plot(traj, cond, col_map, ax_lim, ax_tick, title_str,
                   col_mode='light',
                   line_width=3
                   ):
    # Create figure and 3D axis
    fh_3d, axh_3d = create_3d_axes()

    # Plot
    tmp.plot_traj(
        traj, cond, col_map,
        axh=axh_3d,
        col_mode=col_mode,
        line_width=line_width,
        mode='3d'
    )

    # Format plot (axes, title)
    axh_3d.set_xlim(ax_lim)
    axh_3d.set_ylim(ax_lim)
    axh_3d.set_zlim(ax_lim)
    axh_3d.set_xticks(ax_tick)
    axh_3d.set_yticks(ax_tick)
    axh_3d.set_zticks(ax_tick)
    axh_3d.set_xlabel('Dim. 1 (decoder)')
    axh_3d.set_ylabel('Dim. 2 (decoder)')
    axh_3d.set_zlabel('Dim. 3 (asymmetry)')
    axh_3d.view_init(elev=30, azim=45)
    fh_3d.suptitle(title_str)

    return fh_3d


if __name__ == "__main__":
    main()
