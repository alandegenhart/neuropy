"""Figure 2 flow field analysis script."""

# Import libraries
import os
import sys
import multiprocessing as mp
import argparse

# Define directory paths and import modules
home_dir = os.path.expanduser('~')
src_dir = os.path.join(home_dir, 'src', 'neuropy')
sys.path.append(src_dir)
import neuropy as neu
import neuropy.el.ms.fig_4 as f4
import neuropy.temp as tmp

import pandas as pd
import numpy as np
import scipy


def main():
    """Main analysis function."""

    # TODO: turn these into input arguments
    # Constant parameters -- eventually these will be turned into input args
    location = 'ssd'
    save_dir = os.path.join(home_dir, 'results', 'el_ms', 'fig_2')
    params = {
        'grid_delta': 10,
        'grid_n_min': 2
    }

    # Load experiment log
    EL = neu.el.util.ExperimentLog()
    EL.load()
    EL.get_data_path(location)
    criteria = neu.el.util.get_valid_criteria('two_target_int_rot')
    EL.apply_criteria(criteria)

    # Get experiments -- these will consist of sets of
    task_list = ['tt_int', 'tt_rot']
    experiment_list = EL.get_experiment_sets(task_list)
    # Define data and results locations
    data_path, results_path = neu.el.util.get_data_path(location)

    # Get data paths
    row_num = 0  # TODO: things below here will be wrapped in a function
    subject = experiment_list.iloc[row_num]['subject']
    dataset = str(experiment_list.iloc[row_num]['dataset'])
    int_file = experiment_list.iloc[row_num]['tt_int'][0][0] + '_pandasData.hdf'
    rot_file = experiment_list.iloc[row_num]['tt_rot'][0][0] + '_pandasData.hdf'
    data_dir, _ = os.path.split(experiment_list.iloc[row_num]['dir_path'])

    # TODO: the code below is copied from fig_4.py.  It would be good to make
    #   this into a separate function that would load a dataset and the
    #   associated decoder/GPFA files

    # --- Step 1: Load data, decoding, and GPFA parameters ---
    # Load data
    pandas_dir = os.path.join(data_dir, 'pandasData')
    df_int = pd.read_hdf(os.path.join(pandas_dir, int_file), 'df')
    df_rot = pd.read_hdf(os.path.join(pandas_dir, rot_file), 'df')

    # Drop unused columns to save space
    drop_cols = [
        'acc', 'intTargPos', 'intTargSz', 'spikes', 'tag', 'trialName', 'tube',
        'vel', 'pos'
    ]
    df_int.drop(columns=drop_cols, inplace=True)
    df_rot.drop(columns=drop_cols, inplace=True)

    # Get decoder names
    int_dec_path = os.path.join(
        data_dir, df_int['decoderName'].iloc[0] + '.mat')
    rot_dec_path = os.path.join(
        data_dir, df_rot['decoderName'].iloc[0] + '.mat')

    # Load decoding parameters and GPFA results
    dec_int = neu.util.convertmat.convert_mat(int_dec_path)['bci_params']
    dec_rot = neu.util.convertmat.convert_mat(rot_dec_path)['bci_params']
    neu.el.proc.clean_bci_params(dec_int)
    neu.el.proc.clean_bci_params(dec_rot)

    # Define paths to GPFA data
    gpfa_results_dir = os.path.join(data_dir, 'analysis', 'mat_results')
    int_dec_num = 5  # Need to re-convert data to get this from the params
    rot_dec_num = 10  # Need to re-convert data to get this from the params
    int_gpfa_path = os.path.join(
        gpfa_results_dir,
        'run{:03d}'.format(int_dec_num),
        'gpfa_xDim{}.mat'.format(dec_int['xDim'])
    )
    rot_gpfa_path = os.path.join(
        gpfa_results_dir,
        'run{:03d}'.format(rot_dec_num),
        'gpfa_xDim{}.mat'.format(dec_rot['xDim'])
    )
    int_gpfa_path = os.path.join(gpfa_results_dir, int_gpfa_path)
    rot_gpfa_path = os.path.join(gpfa_results_dir, rot_gpfa_path)

    # Load GPFA data
    gpfa_int = neu.util.convertmat.convert_mat(int_gpfa_path)['estParams']
    gpfa_rot = neu.util.convertmat.convert_mat(rot_gpfa_path)['estParams']
    neu.el.proc.clean_gpfa_params(gpfa_int)
    neu.el.proc.clean_gpfa_params(gpfa_rot)

    # Find onset/offset and add back to dataframe.
    onset_idx_int = df_int.apply(tmp.find_traj_onset,
                                 axis=1, result_type='expand')
    df_int['trajOnsetIdx'] = onset_idx_int['trajOnset']
    df_int['trajOffsetIdx'] = onset_idx_int['trajOffset']
    onset_idx_rot = df_rot.apply(tmp.find_traj_onset,
                                 axis=1, result_type='expand')
    df_rot['trajOnsetIdx'] = onset_idx_rot['trajOnset']
    df_rot['trajOffsetIdx'] = onset_idx_rot['trajOffset']

    # Remove non-paired targets
    df_int, targ_cond_int, targ_info_int = tmp.remove_non_paired_trials(df_int)
    df_rot, targ_cond_rot, targ_info_rot = tmp.remove_non_paired_trials(df_rot)

    # Remove failed trials
    df_int, targ_cond_int = remove_failed_trials(df_int, targ_cond_int)
    df_rot, targ_cond_rot = remove_failed_trials(df_rot, targ_cond_rot)

    # --- Step 2: Extract cursor positions ---
    # Get cursor positions for the 3 conditions to compare
    U_int_actual = extract_cursor_traj(df_int, dec_int)
    U_rot_pred = extract_cursor_traj(df_int, dec_rot)
    U_rot_actual = extract_cursor_traj(df_rot, dec_rot)

    # Collect trajectories and target conditions and plot
    cond_str_list = [
        'Intuitive (actual)', 'Rotated (predicted)', 'Rotated (actual)'
    ]
    traj_list = [U_int_actual, U_rot_pred, U_rot_actual]
    targ_cond_list = [targ_cond_int, targ_cond_int, targ_cond_rot]
    fh, results = plot_trajectories(
        cond_str_list, traj_list, targ_cond_list, params)

    # Set figure title
    title_str = [
        'Subject: {}'.format(subject),
        'Dataset: {}'.format(dataset),
        'Grid voxel size: {} mm'.format(params['grid_delta']),
        'Grid min num.: {}'.format(params['grid_n_min'])
    ]
    fh.text(
        0.05, 1 - 0.05,
        '\n'.join(title_str),
        fontsize=12,
        horizontalalignment='left',
        verticalalignment='top'
    )

    # Save figure
    fig_str = '{}_{}_MappingTrajectories.pdf'.format(subject, dataset)
    fig_name = os.path.join(save_dir, fig_str)
    fh.savefig(fig_name)

    # TODO:
    # 1. Pack up results into a dict
    # 2. Save results
    # 3. Move the above code into a standalone function
    # 4. Update to iterate over experiments
    # 5. Load data across experiments
    # 6. Scatter plot of flow difference magnitude
    # 7. Scatter plot of flow field overlap

    return None


def remove_failed_trials(df, targ_cond):
    """Remove failed trials from dataframe."""

    # Get mask indicating successful trials
    successful_mask = df['successful']

    # Remove failed trials from dataframe
    df = df[successful_mask]

    # Remove failed trials from target cond
    targ_cond = [tc for tc, m in zip(targ_cond, successful_mask) if m]

    return df, targ_cond


def extract_cursor_traj(df, dec_params):
    """Extract cursor trajectories from spike counts."""

    # Get neural trajectories for intuitive mapping
    U = df['decodeSpikeCounts'].apply(
        neu.analysis.gpfa.extract_causal_traj, args=(dec_params,)
    )
    # Convert neural trajectories to cursor positions
    U = U.apply(lambda u: dec_params['W'] @ u + dec_params['c'])

    # Truncate cursor positions
    for i in range(U.shape[0]):
        idx = range(df['trajOnsetIdx'].iloc[i],
                    df['trajOffsetIdx'].iloc[i] + 1)
        U.iloc[i] = U.iloc[i][:, idx]

    return U


def plot_trajectories(cond_str_list, traj_list, targ_cond_list, params):
    """Plot trajectories for intuitive, rotated (predicted), and rotated
    (actual) trial sets.

    """
    # Hard-coded parameters
    center_U = np.array([[0, 0]]).T
    max_dist = 200
    hist_bin_width = 10
    hist_max = 200

    # Setup figure
    fh, axh = tmp.subplot_fixed(
        3, len(traj_list), [300, 300],
        x_margin=[200, 200],
        y_margin=[200, 300])

    # Get trajectory color map and axis limits
    col_map = tmp.define_color_map()
    ax_lim = [-150, 150]
    ax_ticks = [ax_lim[0], 0, ax_lim[1]]

    # Iterate over trajectory sets
    traj_data = zip(cond_str_list, traj_list, targ_cond_list)
    flow_field_list = []
    for idx, traj_data_tuple in enumerate(traj_data):

        # Unpack data and get current axis
        cond_str, traj, targ_cond = traj_data_tuple
        curr_ax = axh[0][idx]

        # Plot trajectories
        tmp.plot_traj(
            traj,
            pd.Series(targ_cond),
            col_map,
            col_mode='light',
            line_width=0.5,
            marker_size=7,
            axh=curr_ax
        )

        # Format plot
        curr_ax.set_title(cond_str)
        curr_ax.set_xlim(ax_lim)
        curr_ax.set_ylim(ax_lim)
        curr_ax.set_xticks(ax_ticks)
        curr_ax.set_yticks(ax_ticks)
        curr_ax.set_xlabel('Cursor position (x)')
        curr_ax.set_ylabel('Cursor position (y)')

        # Fit flow fields
        curr_ax = axh[1][idx]

        # Fit flow field and plot
        F = tmp.FlowField()
        F.fit(traj, params['grid_delta'], center_U, max_dist)
        flow_field_list.append(F)
        F.plot(
            min_n=params['grid_n_min'],
            color='k',
            axh=curr_ax
        )

        # Format plot
        curr_ax.set_title(cond_str)
        curr_ax.set_xlim(ax_lim)
        curr_ax.set_ylim(ax_lim)
        curr_ax.set_xticks(ax_ticks)
        curr_ax.set_yticks(ax_ticks)
        curr_ax.set_xlabel('Cursor position (x)')
        curr_ax.set_ylabel('Cursor position (y)')

    # Compare flow fields
    # int_rot -- intuitive (actual) vs rotated (actual)
    # rot_rot -- rotated (actual) vs rotated (predicted)
    flow_comp_int_rot = tmp.compare_flow_fields(
        flow_field_list[0],
        flow_field_list[2],
        n_min=params['grid_n_min'])
    int_rot_str = [cond_str_list[0], cond_str_list[2]]
    flow_comp_rot_rot = tmp.compare_flow_fields(
        flow_field_list[1],
        flow_field_list[2],
        n_min=params['grid_n_min'])
    rot_rot_str = [cond_str_list[1], cond_str_list[2]]

    # Pack up data into results structure
    results = {
        'int_rot_str': int_rot_str,
        'int_rot_flow_comp': flow_comp_int_rot,
        'rot_rot_str': rot_rot_str,
        'rot_rot_flow_comp': flow_comp_rot_rot,
        'p_val': []
    }

    # Plot flow comparison histogram
    curr_ax = axh[2][0]

    # Define histogram data
    hist_data = [
        results['int_rot_flow_comp']['diff'],
        results['rot_rot_flow_comp']['diff']
    ]
    bins = np.arange(0, hist_max, hist_bin_width)
    hist_labels = ['Int (act) vs Rot (act)', 'Rot (pred) vs Rot (act)']
    hist_col = ['xkcd:grey', 'xkcd:cerulean']

    # Plot histogram
    curr_ax.hist(
        hist_data,
        bins=bins,
        histtype='stepfilled',
        alpha=0.7,
        density=False,
        label=hist_labels,
        color=hist_col
    )

    # Plot median
    y_lim = curr_ax.get_ylim()
    x_int_rot = np.median(results['int_rot_flow_comp']['diff']) * np.ones(2)
    x_rot_rot = np.median(results['rot_rot_flow_comp']['diff']) * np.ones(2)
    curr_ax.plot(x_int_rot, y_lim, color=hist_col[0], linestyle='dashed')
    curr_ax.plot(x_rot_rot, y_lim, color=hist_col[1], linestyle='dashed')

    # Compare distributions
    ranksum_stats = scipy.stats.ranksums(
        results['int_rot_flow_comp']['diff'],
        results['rot_rot_flow_comp']['diff']
    )
    p_val = ranksum_stats.pvalue
    results['p_val'] = p_val

    # Format plot
    title_str = 'Flow field comparison\np = {:0.3e}'.format(p_val)
    curr_ax.set_title(title_str)
    curr_ax.legend()
    curr_ax.set_xlabel('Flow difference magnitude (mm)')
    curr_ax.set_ylabel('Voxels')

    # Remove unused axes
    axh[2][1].remove()
    axh[2][2].remove()

    return fh, results


if __name__ == '__main__':
    main()
