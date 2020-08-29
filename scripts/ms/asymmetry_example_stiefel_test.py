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
    col_map = tmp.define_color_map()
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

    # --- Look at trajectory asymmetries ---

    # Define target pair mapping
    cond_to_pair, pair_to_cond = tmp.define_targ_pair_map()
    targ_pair = [cond_to_pair[tc] for tc in targ_cond]
    uni_targ_pair = pair_to_cond.keys()

    # Get features used in identifying projections
    avg_mode = 'standard'
    M_pairs = []
    for utp in uni_targ_pair:
        pair_mask = [True if tp == utp else False for tp in targ_pair]
        pair_cond_code = [tc for tc, m in zip(targ_cond, pair_mask) if m]
        asym_features = neu.el.comp.get_asymmetry_features(
            U_orth[pair_mask], pair_cond_code)

        # Run stiefel optimization
        O = neu.analysis.stiefel.AsymmetryStandard(data=asym_features)
        S = neu.analysis.stiefel.optimize(O)
        M_pairs.append(S['M'])

        # Plot asymmetry optimization results
        fh = neu.analysis.stiefel.plot(S, O)
        title_str = '{}Asymmetry projection optimization :: {}'.format(
            title_str_base, pair_to_cond[utp])
        fig_name = '{}asym_proj_convergence_{}.pdf'.format(
            fig_name_base, pair_to_cond[utp])
        fh.suptitle(title_str)
        fh.savefig(os.path.join(save_dir, fig_name))

        # Project orthonormalized trajectories into rotated space
        U_asym = U_orth.apply(lambda u: S['M'].T @ u)

        # Plot trajectories
        fh_asym = plot_trajectory_pairs(
            U_asym, pd.Series(targ_cond), targ_pair, avg_mode)
        title_str = '{}Asymmetry trajectories :: {}'.format(
            title_str_base, pair_to_cond[utp])
        fh_asym.suptitle(title_str)
        fig_name = '{}asymmetry_trajectories_{}.pdf'.format(
            fig_name_base, pair_to_cond[utp])
        fh_asym.savefig(os.path.join(save_dir, fig_name))

    # --- Look at joint mapping space ---

    # Run SVD on the merged collection of mappings
    M_merged = np.concatenate(M_pairs, axis=1)
    U_m, s, VH = np.linalg.svd(M_merged, full_matrices=False)
    M_svd = U_m[:, 0:4]  # Take the top 4 latents
    U_svd = U_orth.apply(lambda u: M_svd.T @ u)

    # Plot
    fh_svd = plot_2d_projections(U_svd, pd.Series(targ_cond))
    title_str = '{}Merged asymmetry projections'.format(title_str_base)
    fh_svd.suptitle(title_str)
    fig_name = '{}asymmetry_proj_merged.pdf'.format(fig_name_base)
    fh_svd.savefig(os.path.join(save_dir, fig_name))

    # TODO: plot singular values here?

    # --- Look for organized structure ---

    # Decompose SVD space.  The above plot tends to show that neural activity
    # that is highly correlated with cursor position does not lie within the
    # first 2 dimensions.  We then want to find a 2D projection that is the
    # most correlated with the cursor positions.

    # One way to do this would be to project the asymmetry axes

    # 1. Get orthonormal projection of the intuitive mapping and rotate to
    #    to best align with the workspace
    # 2. Collect all of the asymmetry-defining axes and use SVD to find the
    #    top 2 dimensions that capture the most shared variance of the space

    # Find orthonormal projections of target and asymmetry axes independently
    M_target = [M[:, [1]] for M in M_pairs]
    M_target = np.concatenate(M_target, axis=1)
    M_asymmetry = [M[:, [0]] for M in M_pairs]
    M_asymmetry = np.concatenate(M_asymmetry, axis=1)
    U_t, s_t, VH_t = np.linalg.svd(M_target, full_matrices=False)
    U_a, s_a, VH_a = np.linalg.svd(M_asymmetry, full_matrices=False)

    # Plot singular values
    # Setup figure and axis
    fh, axh = tmp.subplot_fixed(
        1, 1, [300, 300],
        x_margin=[150, 150],
        y_margin=[150, 200]
    )

    # Plot and format
    curr_ax = axh[0][0]
    l1 = curr_ax.plot(s_t, label='Target')
    l2 = curr_ax.plot(s_a, label='Asymmetry')
    curr_ax.set_xlabel('Singular value no.')
    curr_ax.set_ylabel('Value')
    curr_ax.set_title('Singular values')
    curr_ax.legend()

    # Save
    fh.savefig(os.path.join(save_dir, 'decomp_singl_vals.pdf'))

    # Based on the above, it appears (as expected), that ~2 dimensions capture
    # most of the variance in each of these spaces.  So, construct a new space
    # based on these
    M_combined = [U_t[:, 0:2], U_a[:, 0:2]]
    M_combined = np.concatenate(M_combined, axis=1)
    U_svd = U_orth.apply(lambda u: M_combined.T @ u)

    # Plot
    fh_svd = plot_2d_projections(U_svd, pd.Series(targ_cond))
    title_str = '{}Merged asymmetry projections (method 2)'.format(title_str_base)
    fh_svd.suptitle(title_str)
    fig_name = '{}asymmetry_proj_merged_2.pdf'.format(fig_name_base)
    fh_svd.savefig(os.path.join(save_dir, fig_name))

    # TODO: apply rotation to each space independently just to line them up

    # Calculate average and re-center (just for plotting purposes)
    avg_traj, avg_cond = average_trajectories_standard(U_svd, targ_cond)
    traj_mean = avg_traj.apply(lambda u: np.mean(u, axis=1))
    traj_mean = traj_mean.mean()
    traj_mean = np.expand_dims(traj_mean, axis=1)
    avg_traj_recentered = avg_traj.apply(lambda u: u - traj_mean)

    # Create 3d plot - projection 1
    traj_3d = avg_traj_recentered.apply(lambda u: u[(2, 1, 0), :])
    ax_lim = [[-5, 5], [-10, 10], [-10, 10]]
    fh_3d, axh = create_3d_plot(
        traj_3d, avg_cond, col_map, ax_lim, '3D Projection',
        col_mode='dark'
    )
    fh_3d.savefig(os.path.join(save_dir, '3d_proj_1.pdf'))

    # Create 3d plot - projection 2
    traj_3d = avg_traj_recentered.apply(lambda u: u[(3, 1, 0), :])
    ax_lim = [[-5, 5], [-10, 10], [-10, 10]]
    fh_3d, axh = create_3d_plot(
        traj_3d, avg_cond, col_map, ax_lim, '3D Projection',
        col_mode='dark'
    )
    fh_3d.savefig(os.path.join(save_dir, '3d_proj_2.pdf'))

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


def fit_trajectory_model(pos, vel, epochs=1000):
    """Fit trajectory average."""
    input_dim = pos.shape[1]
    units = [20, 20, 10]
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


def average_trajectories(traj, targ_cond, epochs=1000):
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
        model = fit_trajectory_model(traj_trunc.T, traj_diff.T, epochs=epochs)

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
    targ_cond_unique = pd.Series(targ_cond_unique)

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
    cond_unique = pd.Series(cond_unique)

    return traj_avg, cond_unique


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


def create_3d_plot(traj, cond, col_map, ax_lim, title_str,
                   col_mode='light',
                   line_width=3
                   ):
    # Create figure and 3D axis
    fh_3d, axh_3d = create_3d_axes()

    # Plot projections onto each axis.  To do this, the value of of one axis of
    # the trajectories is fixed and the data is plotted (using light/thin
    # colors).

    # Define function to fix the value of an axis
    def fix_axis(u, idx, value):
        # Note -- need to make a copy of u, otherwise it will modify in-place
        import copy
        u_fixed = copy.deepcopy(u)
        u_fixed[idx, :] = value
        return u_fixed

    fixed_ax_val = [ax_lim[0][0], ax_lim[1][1], ax_lim[2][0]]
    for idx, v in enumerate(fixed_ax_val):
        # Fix value of one axis
        traj_fixed = traj.apply(fix_axis, args=(idx, v,))

        # Plot
        tmp.plot_traj(
            traj_fixed, cond, col_map,
            axh=axh_3d,
            col_mode='light',
            line_width=1,
            mode='3d'
        )

    # Plot 3d data
    tmp.plot_traj(
        traj, cond, col_map,
        axh=axh_3d,
        col_mode=col_mode,
        line_width=line_width,
        mode='3d'
    )

    # Format plot (axes, title)
    axh_3d.set_xlim(ax_lim[0])
    axh_3d.set_ylim(ax_lim[1])
    axh_3d.set_zlim(ax_lim[2])
    tick_ax = 0
    ax_tick = [ax_lim[tick_ax][0], np.mean(ax_lim[tick_ax]), ax_lim[tick_ax][1]]
    axh_3d.set_xticks(ax_tick)
    tick_ax = 1
    ax_tick = [ax_lim[tick_ax][0], np.mean(ax_lim[tick_ax]), ax_lim[tick_ax][1]]
    axh_3d.set_yticks(ax_tick)
    tick_ax = 2
    ax_tick = [ax_lim[tick_ax][0], np.mean(ax_lim[tick_ax]), ax_lim[tick_ax][1]]
    axh_3d.set_zticks(ax_tick)
    axh_3d.set_xlabel('Dim. 3 (asymmetry)')
    axh_3d.set_ylabel('Dim. 1 (decoder)')
    axh_3d.set_zlabel('Dim. 2 (decoder)')
    axh_3d.view_init(elev=30, azim=-45)
    fh_3d.suptitle(title_str)

    return fh_3d, axh_3d


def plot_trajectory_pairs(U, targ_cond, set_cond, avg_mode):
    """Plot trajectories for target pairs."""

    # Get unique target pair sets to analyze
    uni_set_cond = set(set_cond)
    cond_to_pair, pair_to_cond = tmp.define_targ_pair_map()

    # Create figure
    n_col = len(uni_set_cond) + 1
    n_row = 1
    fh, axh = tmp.subplot_fixed(
        n_row, n_col, [300, 300],
        x_margin=[150, 150],
        y_margin=[150, 200]
    )
    col_map = tmp.define_color_map()

    # Iterate over target pairs
    x_lim = []
    y_lim = []
    for idx, usc in enumerate(uni_set_cond):
        # Create mask for current target pair
        set_mask = [usc == sc for sc in set_cond]

        # Plot trajectories
        curr_ax = axh[0][idx]
        tmp.plot_traj(
            U[set_mask],
            targ_cond[set_mask],
            col_map,
            axh=curr_ax,
            col_mode='light'
        )

        # Collect axis limits
        x_lim.append(curr_ax.get_xlim())
        y_lim.append(curr_ax.get_ylim())

        # Set title
        curr_ax.set_title('Target pair: {}'.format(pair_to_cond[usc]))

    # Average trajectories
    if avg_mode == 'neuralnetwork':
        avg_traj, avg_cond = average_trajectories(U, targ_cond, epochs=300)
    elif avg_mode == 'standard':
        avg_traj, avg_cond = average_trajectories_standard(U, targ_cond)

    curr_ax = axh[0][idx + 1]
    tmp.plot_traj(
        avg_traj,
        pd.Series(avg_cond),
        col_map,
        axh=curr_ax,
        col_mode='dark',
        line_width=2
    )
    curr_ax.set_title('Average trajectories')

    # Find max/min axis limits
    x_lim = np.stack(x_lim)
    x_lim = [x_lim.min(), x_lim.max()]
    y_lim = np.stack(y_lim)
    y_lim = [y_lim.min(), y_lim.max()]
    for plot_no in range(n_col):
        curr_ax = axh[0][plot_no]
        curr_ax.set_xlim(x_lim)
        curr_ax.set_ylim(y_lim)
        curr_ax.set_aspect('equal')
        curr_ax.set_xlabel('Asymmetry axis')
        curr_ax.set_ylabel('Target axis')

    return fh


def plot_2d_projections(traj, targ_cond):
    """Plot all 2d projections of averaged trajectories."""

    # Average trajectories
    avg_traj, avg_cond = average_trajectories_standard(traj, targ_cond)

    # Setup plot
    m_dim = traj.iloc[0].shape[0]
    proj_dims = list(itertools.combinations(range(m_dim), 2))
    n_col = len(proj_dims)
    n_row = 1
    fh, axh = tmp.subplot_fixed(
        n_row, n_col, [300, 300],
        x_margin=[150, 150],
        y_margin=[150, 200]
    )
    col_map = tmp.define_color_map()

    # Define all dimensions pairs
    for idx, d in enumerate(proj_dims):
        # Get trajectories
        avg_traj_proj = avg_traj.apply(lambda u: u[d, :])

        # Plot
        curr_ax = axh[0][idx]
        tmp.plot_traj(
            avg_traj_proj,
            pd.Series(avg_cond),
            col_map,
            axh=curr_ax,
            col_mode='dark',
            line_width=2
        )

        curr_ax.set_xlabel('SVD dim. {}'.format(d[0]))
        curr_ax.set_ylabel('SVD dim. {}'.format(d[1]))

    return fh


if __name__ == "__main__":
    main()
