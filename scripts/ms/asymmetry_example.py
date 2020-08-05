"""Figure 4 main analysis

"""


# Import libraries
import os
import sys
import itertools
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
    # Get dataset and experiment info, set up directory for saving results
    ds_info, experiment_list = get_dataset_list()
    save_dir = os.path.join(
        os.path.expanduser('~'), 'results', 'el_ms', 'proj_test'
    )
    os.makedirs(save_dir, exist_ok=True)
    fig_name_base = '{}_{}_'.format(ds_info['subject'], ds_info['dataset'])
    title_str_base = '{} {} :: '.format(ds_info['subject'], ds_info['dataset'])

    # Load data
    n_ds = len(experiment_list)
    df_list = []
    targ_cond = []
    for ds_idx in range(n_ds):
        # Load data
        print('Loading dataset {} of {}.'.format(ds_idx+1, n_ds))
        data_dir, _ = os.path.split(experiment_list.iloc[ds_idx]['dir_path'])
        pandas_dir = os.path.join(data_dir, 'pandasData')
        pandas_file = experiment_list.iloc[ds_idx]['tt_int'][0][0]
        file_path = os.path.join(pandas_dir, pandas_file + '_pandasData.hdf')
        df, dec, gpfa_params, targ_cond_ds, targ_info = load_data(
            ds_info, data_dir, file_path)

        # Truncate trajectories
        df = df.apply(truncate_traj, axis=1)
        df_list.append(df)
        targ_cond.extend(targ_cond_ds)

    # Merge datasets
    df = pd.concat(df_list)

    # --- Plot cursor trajectories ---

    # Plot
    col_map = tmp.define_color_map(style='circular')
    fig_title_str = title_str_base + 'Cursor trajectories'
    fh = create_cursor_traj_plot(
        df['decodeState'], targ_cond, col_map, fig_title_str
    )
    fig_name_str = fig_name_base + 'cursorTrajectories.pdf'
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
    #asym_features = neu.el.comp.get_asymmetry_features(U_orth, targ_cond)

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

    # The columns of W_null are orthonormal, and appear to be ordered by shared
    # variance.  Define a new set of weights to map the latents into the
    # [potent, null] joint space
    W_joint = np.concatenate([W_orthnorm, W_null], axis=1)

    # Project neural activity into null and potent spaces
    U_null = U_orth.apply(lambda u: W_null.T @ u)
    U_potent = U_orth.apply(lambda u: W_orthnorm.T @ u)
    U_joint = U_orth.apply(lambda u: W_joint.T @ u)

    # Find condition-invariant response.
    avg_traj, avg_cond = average_trajectories_standard(U_null, targ_cond)
    ci_vecs, ci_vals = find_cond_inv_axis(U_null, targ_cond)

    # Plot eigenvalue spectrum
    fh, axh = tmp.subplot_fixed(
        1, 1, [300, 300],
        x_margin=[150, 150],
        y_margin=[150, 200]
    )
    curr_ax = axh[0][0]
    curr_ax.plot(ci_vals)
    curr_ax.set_xlabel('Eigenvalue #')
    curr_ax.set_ylabel('Value')
    curr_ax.set_title('CI (nullspace) eigenvalue spectrum')
    fh.savefig(os.path.join(save_dir, 'Eig_spectrum.pdf'))

    # Truncate condition-invariant eigenvalues
    n_ci = 1
    ci_vecs = ci_vecs[:, 0:n_ci]  # 8 x 1
    ci_null = sp.linalg.null_space(ci_vecs.T)  # 8 x 7
    W_ci = W_null @ ci_vecs  # 10 x n_ci
    W_ci_null = W_null @ ci_null  # 10 x (8 - n_ci)
    W_decomp = np.concatenate([W_orthnorm, W_ci, W_ci_null], axis=1)

    # Plot decomposition  # TODO: define this function
    fh = plot_trajectory_decomposition(U_orth, W_decomp, targ_cond)
    fh.savefig(os.path.join(save_dir, 'Traj_decomp.pdf'))

    # Run LDA -- this is a sort of "poor man's" projection optimization.  This
    # can work OK, but is suboptimal in that there are no constraints placed on
    # the start/end points.  This means that we won't necessarily get a
    # projection that shows an asymmetry -- we can also get projections where
    # the neural activity is offset along the "asymmetry" axis.
    #X_center_null = W_null.T @ asym_features['x_center']
    #D_lda, J = neu.analysis.math.fisher_lda(
    #    X_center_null, asym_features['cond'])
    #W_lda = W_null @ D_lda
    
    # Method 2 -- optimization-based
    #p = fit_asymmetry_axis(asym_features, W_null)
    #W_lda = W_null @ p

    # Define new set of projection vectors consisting of the intuitive space
    # and 1st LDA dimension
    #W_combined = np.concatenate([W_orthnorm, W_lda], axis=1)
    #U_combined = U_orth.apply(lambda u: W_combined.T @ u)

    # Average trajectories -- limit this to the top 4 dimensions b/c we are
    # fitting a model with a lot of parameters
    U_joint_temp = U_joint.apply(lambda u: u[0:4, :])
    avg_traj, avg_cond = average_trajectories(U_joint_temp, targ_cond)

    # --- Plot 2D null space projections ---
    # It is useful to look at these plots b/c we want to confirm what the neural
    # activity looks like outside of the potent space.

    ax_lim = [-20, 20]
    ax_tick = [ax_lim[0], 0, ax_lim[1]]

    title_str = title_str_base + 'Projections (null space)'
    fh_2d = create_2d_plot(
        U_null, pd.Series(targ_cond), ax_lim, ax_tick, col_map, title_str
    )
    # Save 2D plot
    fig_name_str = fig_name_base + '2D_projections_null.pdf'
    fig_name = os.path.join(save_dir, fig_name_str)
    fh_2d.savefig(fig_name)

    # --- Plot average trajectories ---
    # Here we just plot the potent + null space, rather than the "asymmetry"
    # axis found by LDA.  To help with trajectory averaging, limit the
    # trajectories to the first 4 dimensions.

    # Plot average 2D trajectories
    ax_lim = [-20, 20]  # Note -- this will cut off a lot of the cond-inv resp
    ax_tick = [ax_lim[0], 0, ax_lim[1]]

    title_str = title_str_base + '2D Projections (averaged)'
    dims = list(itertools.combinations(range(4), 2))
    fh_2d = create_2d_plot(
        avg_traj, pd.Series(avg_cond), ax_lim, ax_tick, col_map, title_str,
        dims=dims,
        col_mode='dark',
        line_width=2
    )
    # Save 2D plot
    fig_name_str = fig_name_base + '2D_projections_avg.pdf'
    fig_name = os.path.join(save_dir, fig_name_str)
    fh_2d.savefig(fig_name)

    # Plot 3D average trajectories -- potent + null dim 1
    fig_title_str = fig_name_base + '3D projection (averaged - null 1)'
    fh_3d = create_3d_plot(
        avg_traj, pd.Series(avg_cond), col_map, ax_lim, ax_tick, fig_title_str,
        col_mode='dark',
        line_width=3
    )
    # Save figure
    fig_name_str = fig_name_base + '3D_projections_avg_null_1.pdf'
    fig_name = os.path.join(save_dir, fig_name_str)
    fh_3d.savefig(fig_name)

    # Plot average trajectories -- potent + null dim 2
    avg_traj_temp = avg_traj.apply(lambda u: u[[0, 1, 3], :])
    fig_title_str = title_str_base + '3D projection (averaged - null 2)'
    fh_3d = create_3d_plot(
        avg_traj_temp, pd.Series(avg_cond), col_map, ax_lim, ax_tick, fig_title_str,
        col_mode='dark',
        line_width=3
    )
    # Save figure
    fig_name_str = fig_name_base + '3D_projections_avg_null_2.pdf'
    fig_name = os.path.join(save_dir, fig_name_str)
    fh_3d.savefig(fig_name)
    
    return None


def get_dataset_list():
    """Get a list of datasets to analyze."""
    ds_info = {
        'location': 'ssd',
        'subject': 'Earl',
        'dataset': 20190315,
        'criteria_set': 'two_target_int',
        'dec_num': 5,
        'targ_pair_idx': 0
    }

    # Get valid datasets
    EL = neu.el.util.ExperimentLog()
    EL.load()
    EL.get_data_path(ds_info['location'])
    criteria = neu.el.util.get_valid_criteria(ds_info['criteria_set'])
    criteria['subject'] = ds_info['subject']
    criteria['dataset'] = ds_info['dataset']
    # criteria['gpfa_rot'] = 'cond_4'  # For 20180927
    # criteria['align'] = 'start, possible bug'  # For 20180927
    criteria['gpfa_rot'] = 'na'  # For 20190312
    criteria['align'] = 'start'  # For 20190312
    EL.apply_criteria(criteria)

    # Get experiment sets
    task_list = ['tt_int']
    experiment_list = EL.get_experiment_sets(task_list)

    return ds_info, experiment_list


def load_data(ds_info, data_dir, file_path):
    """Load pandas data for a single dataset."""

    """Step 1: Load data, decoding, and GPFA parameters."""
    # Load data
    df = pd.read_hdf(file_path, 'df')

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
    epochs = 1000
    act_fn = tf.keras.activations.relu
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(units[0], activation=act_fn, input_shape=[input_dim]),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(units[1], activation=act_fn),
            tf.keras.layers.Dropout(0.1),
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


def average_trajectories_standard(traj, cond):
    """Average trajectories.

    This function averages trajectories using a standard time-based averaging
    method.  Trajectories are assumed to be aligned temporally to some initial
    event (cue onset or similar).  The trajectory average will be truncated at
    the time point where there are less than 50% of trials.
    """

    # Get unique target conditions and number of samples to include in the avg
    cond_unique = list(set(cond))
    n_samp = traj.apply(lambda t: t.shape[1])
    n_samp = np.sort(n_samp)
    max_samp = n_samp[np.floor(n_samp.shape[0] / 2).astype(int) + 1]

    # Define function to truncate trajectory
    n_dim = traj.iloc[0].shape[0]
    def trunc_traj(x):
        x_trunc = np.full([n_dim, max_samp], np.nan)
        idx = min(x.shape[1], max_samp)
        x_trunc[:, 0:idx] = x[:, 0:idx]
        return x_trunc

    # Iterate over targets
    traj_avg = []
    for idx, cu in enumerate(cond_unique):
        # Get data for the current condition and average
        cond_mask = [True if c == cu else False for c in cond]
        traj_cond = traj[cond_mask].apply(trunc_traj).to_numpy()
        traj_cond = np.stack(traj_cond)  # now trials x dim x samp
        traj_avg.append(np.nanmean(traj_cond, axis=0))  # dim x samp

    traj_avg = pd.Series(traj_avg)

    return traj_avg, cond_unique

def find_cond_inv_axis(traj, cond):
    """Find condition-invariant axis in neural trajectories."""
    # Average trajectories.  Probably want to average within trial conditions
    # first, and then across conditions in order to prevent bias.
    avg_traj, avg_cond = average_trajectories_standard(traj, cond)
    x = np.nanmean(np.stack(avg_traj), axis=0)  # Avg over conditions

    # Re-center data and calculate covariance
    c = np.mean(x, axis=1, keepdims=True)
    x_c = x - c
    S = np.cov(x_c)

    # Run PCA and look at eigenspectrum
    eig_vals, eig_vecs = np.linalg.eig(S)

    return eig_vecs, eig_vals


def create_cursor_traj_plot(pos, cond, col_map, title_str):
    """Plot cursor trajectories.
    """
    # Setup figure and axis
    fh, axh = tmp.subplot_fixed(
        1, 1, [300, 300],
        x_margin=[150, 150],
        y_margin=[150, 200]
    )

    # Average trajectories
    avg_pos, avg_cond = average_trajectories_standard(pos, cond)

    # Plot -- all trials
    curr_ax = axh[0][0]
    tmp.plot_traj(
        pos,
        pd.Series(cond),
        col_map,
        axh=curr_ax,
        col_mode='light'
    )

    # Plot - avg. trajectories
    tmp.plot_traj(
        avg_pos,
        pd.Series(avg_cond),
        col_map,
        axh=curr_ax,
        col_mode='dark',
        line_width=3
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


def create_2d_plot(traj, cond, ax_lim, ax_tick, col_map, title_str,
                   dims=None, col_mode='light', line_width=1):
    """Create 2d projection plots."""

    # Setup axes and axis limits
    if dims is None:
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
            col_mode=col_mode,
            line_width=line_width
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


def plot_trajectory_decomposition(U, W, cond):
    """Create plots decomposing neural trajectories."""

    # Set up figure
    fh, axh = tmp.subplot_fixed(
        2, 4, [300, 300],
        x_margin=[150, 150],
        y_margin=[150, 200]
    )
    col_map = tmp.define_color_map(style='circular')

    # Apply transformation to latents
    U_decomp = U.apply(lambda u: W.T @ u)
    U_decomp_avg, avg_cond = average_trajectories_standard(U_decomp, cond)

    # Subplot 1: 2D potent space (latent dims. 1 and 2)
    U_plot = U_decomp_avg.apply(lambda u: u[0:2, :])
    curr_ax = axh[0][0]
    tmp.plot_traj(
        U_plot,
        pd.Series(avg_cond),
        col_map,
        axh=curr_ax,
        col_mode='dark',
        line_width=2
    )
    #curr_ax.set_xlim(ax_lim)
    #curr_ax.set_ylim(ax_lim)
    #curr_ax.set_xticks(ax_tick)
    #curr_ax.set_yticks(ax_tick)
    curr_ax.set_xlabel('Cursor dim. 1')
    curr_ax.set_ylabel('Cursor dim. 2')
    curr_ax.set_title('Potent (cursor) space')

    # Subplot 2: 1D plot (vs time) of the condition-invariant response
    U_plot = U_decomp_avg.apply(lambda u: u[2:3, :])
    curr_ax = axh[0][1]
    for uni_cond, u in zip(avg_cond, U_plot):
        curr_ax.plot(u[0, :], color=col_map[uni_cond]['dark'])

    curr_ax.set_xlabel('Time (45ms bin)')
    curr_ax.set_ylabel('Cond. inv. activity')
    curr_ax.set_title('Condition-invariant response')

    # Subplot 3: 2D plot of null dims. 1 and 2
    U_plot = U_decomp_avg.apply(lambda u: u[3:5, :])
    curr_ax = axh[0][2]
    tmp.plot_traj(
        U_plot,
        pd.Series(avg_cond),
        col_map,
        axh=curr_ax,
        col_mode='dark',
        line_width=2
    )
    #curr_ax.set_xlim(ax_lim)
    #curr_ax.set_ylim(ax_lim)
    #curr_ax.set_xticks(ax_tick)
    #curr_ax.set_yticks(ax_tick)
    curr_ax.set_xlabel('Null dim. 1')
    curr_ax.set_ylabel('Null dim. 2')
    curr_ax.set_title('Null space')

    # Subplot 4: 2D plot of null dims. 3 and 4
    U_plot = U_decomp_avg.apply(lambda u: u[5:7, :])
    curr_ax = axh[0][3]
    tmp.plot_traj(
        U_plot,
        pd.Series(avg_cond),
        col_map,
        axh=curr_ax,
        col_mode='dark',
        line_width=2
    )
    # curr_ax.set_xlim(ax_lim)
    # curr_ax.set_ylim(ax_lim)
    # curr_ax.set_xticks(ax_tick)
    # curr_ax.set_yticks(ax_tick)
    curr_ax.set_xlabel('Null dim. 3')
    curr_ax.set_ylabel('Null dim. 4')
    curr_ax.set_title('Null space')

    # Plot timecourses of top 4 null dimensions -- mainly to find the point of
    # peak separation
    for col in range(4):
        curr_ax = axh[1][col]
        null_idx = col + 3
        U_plot = U_decomp_avg.apply(lambda u: u[null_idx, :])
        # Iterate over conditions
        for uni_cond, u in zip(avg_cond, U_plot):
            curr_ax.plot(u, color=col_map[uni_cond]['dark'])

        curr_ax.set_xlabel('Time (45ms bin)')
        curr_ax.set_ylabel('Null dim. {}'.format(col + 1))
        curr_ax.set_title('Null space time course')

    return fh


if __name__ == "__main__":
    main()
