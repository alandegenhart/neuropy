# Setup -- Import modules

# Standard modules
import os
import sys
import numpy as np
import scipy
from scipy import stats
import pandas as pd
import matplotlib as mpl

# Custom modules
home_dir = os.path.expanduser('~')
src_dir = os.path.join(home_dir, 'src')
sys.path.append(src_dir)
import neuropy as neu
from neuropy.analysis import gpfa
import neuropy.temp as tmp


def define_param_sets(params):
    """Define all possible parameter sets. """

    import itertools

    # Get all possible permutations of the input parameter values
    p = itertools.product(*params.values())
    param_sets = [
        {k: v for k, v in zip(params.keys(), vals)}
        for vals in list(p)
    ]

    return param_sets


def flow_analysis(subject, dataset, params, base_dir):
    """10D flow analysis.

    """

    """Step 1: Load data, decoding, and GPFA parameters."""

    # Define data paths
    data_dir = os.path.join(
        base_dir, subject, dataset[0:4], dataset[4:6], dataset
    )
    save_dir = os.path.join(data_dir, 'translated', 'pandasData')
    dataset_name = [
        'Earl20180927_04_condGridTask_01_SI_exportData.hdf',
        'Earl20180927_05_twoTargetABBA_rotated_01_SI_exportData.hdf'
    ]

    # Load data
    df_int = pd.read_hdf(os.path.join(save_dir, dataset_name[0]), 'df')
    df_rot = pd.read_hdf(os.path.join(save_dir, dataset_name[1]), 'df')

    # Drop unused columns to save space
    drop_cols = [
        'acc', 'intTargPos', 'intTargSz', 'spikes', 'tag', 'trialName', 'tube',
        'vel', 'pos'
    ]
    df_int.drop(columns=drop_cols, inplace=True)
    df_rot.drop(columns=drop_cols, inplace=True)

    # Get decoder names
    int_dec_path = os.path.join(data_dir,
                                df_int['decoderName'].iloc[0] + '.mat')
    rot_dec_path = os.path.join(data_dir,
                                df_rot['decoderName'].iloc[0] + '.mat')

    # Load decoding parameters and GPFA results
    dec_int = neu.util.convertmat.convert_mat(int_dec_path)['bci_params']
    dec_rot = neu.util.convertmat.convert_mat(rot_dec_path)['bci_params']
    neu.el.proc.clean_bci_params(dec_int)
    neu.el.proc.clean_bci_params(dec_rot)

    # Define paths to GPFA data
    gpfa_results_dir = os.path.join(data_dir, 'analysis', 'mat_results')
    int_dec_num = 5  # Need to re-convert data to get this from the params
    rot_dec_num = 10 # Need to re-convert data to get this from the params
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

    """Step 2: Extract neural trajectories."""

    # Orthonormalize
    C_orth, T, s, VH = gpfa.orthogonalize(gpfa_int['C'])
    S = np.diag(s)  # Diagonal matrix
    total_shared_var = s.sum()
    x_dim = T.shape[0]

    # Get neural trajectories for intuitive mapping
    U_int = df_int['decodeSpikeCounts'].apply(
        neu.analysis.gpfa.extract_causal_traj, args=(dec_int,)
    )
    # Get neural trajectories for rotated mapping -- use the intuitive mapping
    # decoding parameters
    U_rot = df_rot['decodeSpikeCounts'].apply(
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

    # Truncate rotated trajectories
    for i in range(U_rot.shape[0]):
        idx = range(df_rot['trajOnsetIdx'].iloc[i],
                    df_rot['trajOffsetIdx'].iloc[i] + 1)
        U_rot.iloc[i] = U_rot.iloc[i][:, idx]

    # Transform to orthonormalized latents
    U = {'int': U_int, 'rot': U_rot}
    U_orth = {dec: U_dec.apply(lambda u: T @ u) for dec, U_dec in U.items()}

    # Get unique targets to plot. Should only have to do this for the intuitive
    # trials b/c both the intuitive and rotated use the same target config.
    targ_cond_unique = set(targ_cond_int)

    # Check parameters. If the projection mode is 'orth', the maximum number of
    # projections and plotted projections is half the dimensionality
    if params['projection_mode'] == 'orth':
        n_orth_proj_max = int(x_dim/2)
        params['n_proj'] = min(params['n_proj'], n_orth_proj_max)
        params['n_proj_plots'] = min(params['n_proj_plots'], n_orth_proj_max)

    # Before iterating over projections, first get the sets of indices for
    # each dataset (intuitive and rotated)
    # Create a dict with the various combinations
    trial_idx = {'targ': [], 'dec': [], 'idx': [], 'n_trials': [], 'seq': []}
    targ_cond_all = {'int': targ_cond_int, 'rot': targ_cond_rot}
    dec_cond = ['int', 'rot']  # Decoder conditions
    targ_cond = list(targ_cond_unique)  # Unique target conditions
    seq_cond = ['normal', 'reversed']
    comp_cond = []  # List of comparison conditions to evaluate
    for row, tcu in enumerate(targ_cond):
        for dec in dec_cond:
            # Get mask for current dataset
            mask_tc_dec = [True if tc == tcu else False
                           for tc in targ_cond_all[dec]]
            idx_tc_dec = np.argwhere(mask_tc_dec)[:, 0]  # shape: (n_trials, )

            # Iterate over sequence orders
            for s in seq_cond:
                # Append to dictionary
                trial_idx['targ'].append(tcu)
                trial_idx['dec'].append(dec)
                trial_idx['idx'].append(idx_tc_dec)
                trial_idx['n_trials'].append(len(idx_tc_dec))
                trial_idx['seq'].append(s)

        comp_cond.append(define_comparison_conditions(targ_cond, tcu))

    # Determine the number or trials to use for fitting a flow field. Since
    # we're doing a 50/50 split, this will be the minimum across the various
    # datasets/target combinations divided by 2 and rounded down.
    n_split = 2
    n_trials_split = (np.min(trial_idx['n_trials'])/n_split).astype(int)

    # Covert trial index dictionary to a dataframe. This will make it easier to
    # iterate over the rows. Also perform other initialization procedures.
    trial_idx = pd.DataFrame(trial_idx)
    flow_cols = ['targ', 'dec', 'trial_split', 'seq', 'U', 'flow']
    rng = np.random.default_rng()

    # Initialize results. The results dict will have lists with values for
    # each projection. In the case of the 'flow_diff' and 'n_overlap' fields,
    # these will be sub-dicts with fields for each flow comparison (e.g., int vs
    # rot). The values for these fields will then be lists of lists, where each
    # top-level list contains the results for each projection for the comparison
    # of interest, and the sub-list contains the results for each target pair.
    cc_names = [cc['name'] for cc in comp_cond[0]]  # Names of comparison conds
    results = {
        'targ': [],
        'proj_var': [],
        'flow_diff': {n: [] for n in cc_names},
        'n_overlap': {n: [] for n in cc_names}
    }
    # Iterate over projections
    flow_ex = {
        'flow': [],
        'diff_results': [],
        'hist_data': []
    }  # Example flow fields for a single projection/permutation
    for proj in range(params['n_proj']):
        # Display status
        print('Projection: {} of {}'.format(proj + 1, params['n_proj']))

        # Define projection
        if params['projection_mode'] == 'orth':
            # Create identity vector
            P = np.zeros((x_dim, 2))
            dim = proj * 2
            P[dim, 0] = 1
            P[dim + 1, 1] = 1
        elif params['projection_mode'] == 'random':
            P = scipy.linalg.orth(np.random.randn(x_dim, 2))

        # Calculate shared variance -- Note: this is for the projection, meaning
        # that it will be the same for each target condition.
        proj_shared_var = np.trace(P.T @ S @ P)
        frac_shared_var = proj_shared_var/total_shared_var
        results['proj_var'].append(frac_shared_var)

        # Apply projection to data
        U_proj = {dec: u_orth.apply(lambda U: P.T @ U)
                  for dec, u_orth in U_orth.items()}

        # Find data range in the current projection. This is used to set the
        # limits of the flow field. Note that this is across all target
        # conditions, such that comparisons can be made across target
        # conditions using the same grid.
        U_proj_concat = [np.concatenate(U_proj[dec].to_numpy(), axis=1)
                         for dec in dec_cond]
        U_proj_concat = np.concatenate(U_proj_concat, axis=1)
        max_U = U_proj_concat.max(axis=1, keepdims=True)
        min_U = U_proj_concat.min(axis=1, keepdims=True)
        lim_U = np.concatenate([min_U, max_U], axis=1)
        center_U = lim_U.mean(axis=1, keepdims=True)
        max_dist = (lim_U - center_U)[:, 1].max() + params['grid_delta']

        # Create structure for the results for the current projection. For each
        # target/comparison condition, create an empty list to store the data
        results_proj = [
            {
                'targ': tcu,
                'flow_diff': {n: [] for n in cc_names},
                'n_overlap': {n: [] for n in cc_names}
            }
            for tcu in targ_cond
        ]
        for p in range(params['n_permute']):

            # Initialize data frame to store flow fields for the current
            # permutation
            flow_df = {
                'targ': [],
                'dec': [],
                'trial_split': [],
                'seq': [],
                'U': [],
                'flow': []
            }

            # Iterate over rows in the set of trial indices
            for row in trial_idx.itertuples(index=False):

                # Permute trial indices
                row_idx = row.idx
                rnd_idx = rng.permutation(row_idx.shape[0])
                row_idx = row_idx[rnd_idx]  # Shuffle indices

                # Iterate over splits
                for i in range(n_split):
                    # Get data indices
                    onset_idx = i * n_trials_split
                    offset_idx = onset_idx + n_trials_split
                    idx_split = row_idx[onset_idx:offset_idx]
                    U_temp = U_proj[row.dec].iloc[idx_split]

                    # Check to see if the temporal order of the trajectories
                    # should be reversed
                    if row.seq == 'reversed':
                        U_temp = reverse_traj(U_temp)

                    # Fit flow field
                    F = tmp.FlowField()
                    F.fit(U_temp, params['grid_delta'], center_U, max_dist)

                    # Add data to dataframe
                    flow_df['targ'].append(row.targ)
                    flow_df['dec'].append(row.dec)
                    flow_df['trial_split'].append(i)
                    flow_df['seq'].append(row.seq)
                    flow_df['U'].append(U_temp)
                    flow_df['flow'].append(F)

            # Convert dict to dataframe
            flow_df = pd.DataFrame(flow_df)

            # There will now be a set of flow fields for each targ/dec/split
            # condition. Now we need to make the appropriate comparisons
            comp_results_perm = []  # Comparison results for current permutation
            # Iterate over targets
            for targ_idx, cc_targ in enumerate(comp_cond):
                # Iterate over conditions for a given target pair
                comp_results_targ = []  # Results for the current target
                for cc in cc_targ:
                    # Iterate over comparison elements
                    cc_targ_mask = []
                    for i in range(2):
                        # Get the appropriate element from the flow fields
                        mask = (
                            (flow_df['targ'] == cc['targ'][i])
                            & (flow_df['dec'] == cc['dec'][i])
                            & (flow_df['trial_split'] == cc['split'][i])
                            & (flow_df['seq'] == cc['seq'][i])
                        )
                        cc_targ_mask.append(mask)

                    # Compare flow fields and add to results list
                    diff = tmp.compare_flow_fields(
                        flow_df[cc_targ_mask[0]]['flow'].iloc[0],
                        flow_df[cc_targ_mask[1]]['flow'].iloc[0],
                        n_min=params['grid_n_min'])

                    # Append results to list for current target pair. Keep track
                    # of the mask and the diff results
                    comp_results_targ.append(
                        {'mask': cc_targ_mask, 'diff': diff})
                    results_proj[targ_idx]['flow_diff'][cc['name']].append(
                        diff['diff']
                    )
                    results_proj[targ_idx]['n_overlap'][cc['name']].append(
                        diff['n_overlap']
                    )

                # Append results for current target pair to the results for the
                # current permutation
                comp_results_perm.append(comp_results_targ)

            # If this is the first permutation, save results for plotting
            # purposes. Only do this for the first n_proj_plots projections.
            if (p == 0) and (proj < params['n_proj_plots']):
                flow_ex['flow'].append(flow_df)
                flow_ex['diff_results'].append(comp_results_perm)

        # Iterate over targets/comparison conditions and concatenate all flow
        # difference values (diff and overlap)
        hist_data = {
            'proj': proj,
            'proj_var': frac_shared_var,
            'targ': [],
            'flow_diff': {n: [] for n in cc_names},
            'n_overlap': {n: [] for n in cc_names}
        }
        flow_diff_median = {n: [] for n in cc_names}
        n_overlap_mean = {n: [] for n in cc_names}
        for targ_idx, cc_targ in enumerate(comp_cond):
            for cc in cc_targ:
                # Concatenate flow difference across permutations
                flow_diff_concat = np.concatenate(
                    results_proj[targ_idx]['flow_diff'][cc['name']]
                )
                # Get overlap for all permutations
                n_overlap = np.array(
                    results_proj[targ_idx]['n_overlap'][cc['name']]
                )

                # Remove any NaNs
                nan_mask = np.isnan(flow_diff_concat)
                flow_diff_concat = flow_diff_concat[~nan_mask]
                nan_mask = np.isnan(n_overlap)
                n_overlap = n_overlap[~nan_mask]

                # Add to hist_data results. This should mean each field in
                # 'flow_diff' and 'n_overlap' will have 2 elements -- one for
                # each target.
                hist_data['flow_diff'][cc['name']].append(flow_diff_concat)
                hist_data['n_overlap'][cc['name']].append(n_overlap)

                # Calculate summary statistics for distributions
                flow_diff_median[cc['name']].append(np.median(flow_diff_concat))
                n_overlap_mean[cc['name']].append(np.mean(flow_diff_concat))

            # Add target name to hist data dict
            hist_data['targ'].append(cc_targ[0]['targ'][0])

        # Add summary statistics to results structure
        results['targ'].append(hist_data['targ'])
        for ccn in cc_names:
            results['flow_diff'][ccn].append(flow_diff_median[ccn])
            results['n_overlap'][ccn].append(n_overlap_mean[ccn])

        # Add hist data for current projection to flow examples dict
        if proj < params['n_proj_plots']:
            flow_ex['hist_data'].append(hist_data)

    # TODO: Fix targets (not getting the appropriate names)

    # --- Now we should have all of the data for the current projection ---
    # NOTE: the one thing that is not being done here is any type of stats
    # test. We can leave that out for now, as the final statistical test
    # will probably be some sort of summary across projections

    return results, flow_ex, comp_cond


def reverse_traj(U):
    """Reverse temporal order of trajectory."""
    # Reverse elements in trajectory.
    return U.apply(lambda u: np.flip(u, axis=1))


def define_comparison_conditions(targ_cond, tcu):
    """Define list of flow field comparisons."""
    comp_cond = []
    # Comparision -- int vs rot
    cc = {
        'name': 'int_vs_rot',
        'targ': [tcu, tcu],
        'dec': ['int', 'rot'],
        'split': [0, 0],
        'seq': ['normal', 'normal']
    }
    comp_cond.append(cc)
    # Comparision -- int vs int
    cc = {
        'name': 'int_vs_int',
        'targ': [tcu, tcu],
        'dec': ['int', 'int'],
        'split': [0, 1],
        'seq': ['normal', 'normal']
    }
    comp_cond.append(cc)
    # Comparision -- rot vs rot
    cc = {
        'name': 'rot_vs_rot',
        'targ': [tcu, tcu],
        'dec': ['rot', 'rot'],
        'split': [0, 1],
        'seq': ['normal', 'normal']
    }
    comp_cond.append(cc)
    # Comparision -- int vs int (reversed)
    cc = {
        'name': 'int_vs_int_rev',
        'targ': [tcu, tcu],
        'dec': ['int', 'int'],
        'split': [0, 1],
        'seq': ['normal', 'reversed']
    }
    comp_cond.append(cc)
    # Comparision -- rot vs rot (reversed)
    cc = {
        'name': 'rot_vs_rot_rev',
        'targ': [tcu, tcu],
        'dec': ['rot', 'rot'],
        'split': [0, 1],
        'seq': ['normal', 'reversed']
    }
    comp_cond.append(cc)
    # Comparision -- int vs int (other target pair)
    cc = {
        'name': 'int_vs_int_alt',
        'targ': [tcu, list(set(targ_cond) - {tcu})[0]],
        'dec': ['int', 'int'],
        'split': [0, 1],
        'seq': ['normal', 'normal']
    }
    comp_cond.append(cc)
    # Comparision -- rot vs rot (other target pair)
    cc = {
        'name': 'rot_vs_rot_alt',
        'targ': [tcu, list(set(targ_cond) - {tcu})[0]],
        'dec': ['rot', 'rot'],
        'split': [0, 1],
        'seq': ['normal', 'normal']
    }
    comp_cond.append(cc)

    return comp_cond


def plot_flow_ex(subject, dataset, flow_ex, comp_cond, params):
    """Create example flow field plots."""

    # Set up directory for saving results
    home_dir = os.path.expanduser('~')
    save_dir_base = os.path.join(
        home_dir, 'results', 'el_ms', 'fig_4', 'flow_10D')
    params_str = [
        subject,
        dataset,
        'projMode_{}'.format(params['projection_mode']),
        'nProj_{}'.format(params['n_proj']),
        'nPerm_{}'.format(params['n_permute']),
        'gridDelta_{}'.format(params['grid_delta']),
        'gridNMin_{}'.format(params['grid_n_min'])
    ]
    params_str = '_'.join(params_str)
    save_dir = os.path.join(save_dir_base, params_str)
    os.makedirs(save_dir, exist_ok=True)

    # Define colormap (used for plotting)
    col_map = tmp.define_color_map()
    line_col = {
        'int': 'xkcd:medium blue',
        'rot': 'xkcd:blood red',
        'introt': 'xkcd:gold'
    }

    # Define new flow colors
    targ = flow_ex['hist_data'][0]['targ']
    col_shift = [-0.05, 0.05]
    col_flow = {}
    for t in targ:
        col_hex = col_map[t]['dark']
        col_rgb = mpl.colors.to_rgb(col_hex)
        col_hsv = mpl.colors.rgb_to_hsv(col_rgb)

        col_new = []
        for cs in col_shift:
            # Add shift
            col_temp = col_hsv + np.array([cs, 0, 0])
            # If the shift exceeds the valid range, correct
            if col_temp[0] < 0:
                col_temp[0] += 1
            elif col_temp[0] > 1:
                col_temp[0] += -1
            # Convert back to hex
            col_rgb = mpl.colors.hsv_to_rgb(col_temp)
            col_new.append(mpl.colors.to_hex(col_rgb))

        # Add color
        col_flow[t] = col_new

    # Iterate over examples
    for fd, dd, hd in zip(
            flow_ex['flow'], flow_ex['diff_results'], flow_ex['hist_data']):
        # Data organization:
        #   fd -- DataFrame with flow field and trajectory information
        #   dd -- List of lists with flow comparison information:
        #   dd -- (2, ) list of flow diff (for each target)
        #   dd[i] -- (7, ) list of flow diff results for a target i
        #   dd[i][j] -- dict with flow results for target i, condition j
        #   dd[i][j]['mask'] -- (2, ) list of masks for each flow field
        #   dd[i][j]['diff']['diff'] -- vector of flow differences

        # Iterate over targets
        for targ_idx, targ_diff in enumerate(dd):
            targ = hd['targ'][targ_idx]

            # Setup figure
            fh, axh = tmp.subplot_fixed(
                len(targ_diff), 5, [300, 300],
                x_margin=[200, 200],
                y_margin=[200, 300])

            # Iterate over comparisons
            for cc_idx, cc in enumerate(targ_diff):

                # Get specific condition
                cond = comp_cond[targ_idx][cc_idx]

                # Iterate over flow fields
                for flow_idx, flow_mask in enumerate(cc['mask']):

                    # Get flow field and trajectories
                    F = fd[flow_mask]['flow'].iloc[0]
                    U = fd[flow_mask]['U'].iloc[0]

                    # Setup plot for flow field
                    curr_ax = axh[cc_idx][flow_idx]

                    # Get axis limits from the specified grid.
                    x_lim = F.grid['x'][[0, -1]]
                    y_lim = F.grid['y'][[0, -1]]

                    # Plot flow
                    F.plot(
                        min_n=params['grid_n_min'],
                        color=col_flow[cond['targ'][flow_idx]][flow_idx],
                        axh=curr_ax
                    )

                    # Plot trajectories
                    targ_cond_series = pd.Series(
                        [cond['targ'][flow_idx]] * U.shape[0])
                    tmp.plot_traj(
                        U,
                        targ_cond_series,
                        col_map,
                        col_mode='light',
                        line_width=0.5,
                        marker_size=7,
                        axh=curr_ax
                    )
                    curr_ax.set_xlim(x_lim)
                    curr_ax.set_ylim(y_lim)
                    curr_ax.set_title(
                        'Targ: {}, Dec: {}\nSplit: {}, Seq: {}'.format(
                            cond['targ'][flow_idx],
                            cond['dec'][flow_idx],
                            cond['split'][flow_idx],
                            cond['seq'][flow_idx]
                        )
                    )

                    # Plot flow field (on separate plot)
                    curr_ax = axh[cc_idx][2]
                    F.plot(
                        min_n=params['grid_n_min'],
                        color=col_flow[cond['targ'][flow_idx]][flow_idx],
                        axh=curr_ax
                    )

                # Format flow field comparison plot
                curr_ax.set_xlim(x_lim)
                curr_ax.set_ylim(y_lim)
                curr_ax.set_title(
                    'Flow field comparision:\n{}'.format(cond['name'])
                )

                # --- Plotting of histograms goes here ---

            # Set figure title
            title_str = [
                'Subject: {}'.format(subject),
                'Dataset: {}'.format(dataset),
                'Projection mode: {}'.format(params['projection_mode']),
                'Projection num: {}'.format(hd['proj']),
                'Target: {}'.format(targ),
                'Grid spacing: {}'.format(params['grid_delta']),
                'Grid min # overlap: {}'.format(params['grid_n_min']),
                'Projection % shared variance: {:2.2f}'.format(
                    100 * hd['proj_var'])
            ]
            fh.text(
                0.01, 1 - 0.01,
                '\n'.join(title_str),
                fontsize=12,
                horizontalalignment='left',
                verticalalignment='top'
            )

            # Save figure
            fig_str = [
                'FlowComp',
                subject,
                dataset,
                params['projection_mode'],
                '{:03d}'.format(hd['proj']),
                targ
            ]
            fig_str = '_'.join(fig_str) + '.pdf'
            fig_name = os.path.join(save_dir, fig_str)
            fh.savefig(fig_name)

    # Anything else that needs to be done here?
    return None


    # Plot Set 1: flow fields with trajectories
    # First, iterate over comparisons and plot flow field examples. These will
    # consist of 3 plots per comparison condition -- the flow field +
    # trajectories for each condition (plots 1 and 2), as well as the 'overlap'
    # flow field, which shows the flow vectors for the two conditions on the
    # same plot without trajectories.

    # Plot Set 2: histograms
    # Here, we want to plot the distributions of the flow difference and
    # overlap distributions for each comparision.  For ease of visualization,
    # it would probably be best if these histograms all were with respect to the
    # int vs rot distribution, and had the same axis limits and bin sizes

def plot_flow_results(results, flow_ex):
    # --- Plot results for current target condition ---
    """

    # Define colormap (used for plotting)
    col_map = tmp.define_color_map()
    line_col = {
        'int': 'xkcd:medium blue',
        'rot': 'xkcd:blood red',
        'introt': 'xkcd:gold'
    }

    # Check to see if the projection is to be plotted
    if proj >= params['n_proj_plots']:
        continue

    # Get axis limits from the specified grid.
    x_lim = F_int_1.grid['x'][[0, -1]]
    y_lim = F_int_1.grid['y'][[0, -1]]

    # --- Subplot 1: Intuitive flow field ---
    curr_ax = axh[row][0]

    # Plot intuitive flow field
    F_int_1.plot(
        min_n=params['grid_n_min'],
        color=col_map[tcu]['dark'],
        axh=curr_ax
    )

    # Plot intuitive trajectories -- currently using the last random
    # permutation
    targ_cond_series = pd.Series([tcu] * n_trials)
    tmp.plot_traj(
        U_perm_int_1,
        targ_cond_series,
        col_map,
        col_mode='light',
        line_width=0.5,
        marker_size=7,
        axh=curr_ax
    )
    curr_ax.set_xlim(x_lim)
    curr_ax.set_ylim(y_lim)
    curr_ax.set_title(
        'Proj: {}, Targ: {}, Dec: {}'.format(proj, tcu, 'intuitive'))

    # --- Subplot 2: Rotated flow field ---
    curr_ax = axh[row][1]

    # Plot rotated flow field
    F_rot_1.plot(
        min_n=params['grid_n_min'],
        color=col_map[tcu]['dark'],
        axh=curr_ax
    )

    # Plot rotated trajectories -- currently using the last random
    # permutation
    tmp.plot_traj(
        U_perm_rot_1,
        targ_cond_series,
        col_map,
        col_mode='light',
        line_width=0.5,
        marker_size=7,
        axh=curr_ax
    )
    curr_ax.set_xlim(x_lim)
    curr_ax.set_ylim(y_lim)
    curr_ax.set_title(
        'Proj: {}, Targ: {}, Dec: {}'.format(proj, tcu, 'rotated'))

    # --- Subplot 3: Difference heat map ---
    curr_ax = axh[row][2]

    # Set extent for image
    curr_ax.imshow(
        diff_introt_1['diff_grid'].T,
        cmap=plt.get_cmap('Greys'),
        origin='lower',
        extent=np.concatenate([x_lim, y_lim])
    )
    curr_ax.set_xlim(x_lim)
    curr_ax.set_ylim(y_lim)
    curr_ax.set_title('Overlapping voxels - int vs rot')

    # --- Subplot 4: Histogram of flow field differences ---
    curr_ax = axh[row][3]

    # Plot histogram -- intuitive vs intuitive
    hist_data = [
        diff_int_concat,
        diff_rot_concat,
        diff_introt_concat
    ]
    hist_labels = ['int vs int', 'rot vs rot', 'int vs rot']
    hist_col = [line_col['int'], line_col['rot'], line_col['introt']]
    curr_ax.hist(
        hist_data,
        bins=20,
        histtype='stepfilled',
        alpha=0.5,
        density=True,
        label=hist_labels,
        color=hist_col)

    # Plot median values
    y_lim = curr_ax.get_ylim()
    curr_ax.plot(
        np.median(np.concatenate(diff_int_all)) * np.ones((2, )),
        y_lim,
        linestyle='--',
        color=line_col['int']
    )
    curr_ax.plot(
        np.median(np.concatenate(diff_rot_all)) * np.ones((2, )),
        y_lim,
        linestyle='--',
        color=line_col['rot']
    )
    curr_ax.plot(
        np.median(np.concatenate(diff_introt_all)) * np.ones((2, )),
        y_lim,
        linestyle='--',
        color=line_col['introt']
    )

    # Format plot
    curr_ax.legend()
    curr_ax.set_xlabel('Flow difference magnitude')
    curr_ax.set_ylabel('Density')

    # --- Subplot 5: median + CI ---
    curr_ax = axh[row][4]

    # Create box plots using the built-in matplotlib function
    bplot = curr_ax.boxplot(
        hist_data,
        notch=True,
        whis=[2.5, 97.5],
        bootstrap=1000,
        labels=hist_labels,
        patch_artist=True,  # Needed to fill with color
        medianprops={'color':'black'},
        showfliers=False  # Don't show outliers
    )

    # Format plot -- set box colors
    for p, c in zip(bplot['boxes'], line_col.values()):
        p.set_facecolor(c)
        p.set_alpha(0.7)

    curr_ax.set_ylabel('Flow difference magnitude')
    title_str = [
        'Int vs IntRot: p = {:1.3e}'.format(stats_int_introt.pvalue),
        'Rot vs IntRot: p = {:1.3e}'.format(stats_rot_introt.pvalue)
    ]
    # If either of the statistical test was significant, plot the title
    # in bold green
    if ((stats_int_introt.pvalue <= 0.05)
            or (stats_int_introt.pvalue <= 0.05)):
        font_weight = 'bold'
        font_color = 'xkcd:emerald'
    else:
        font_weight = 'normal'
        font_color = 'black'

    curr_ax.set_title(
        '\n'.join(title_str),
        fontdict={'fontweight': font_weight, 'color': font_color}
    )

    # --- Format/save figure (only if plotting the current projection ---

    # Continue to next iteration if not plotting
    if proj >= params['n_proj_plots']:
        continue

    # Set figure title
    title_str = [
        'Subject: {}'.format(subject),
        'Dataset: {}'.format(dataset),
        'Projection mode: {}'.format(params['projection_mode']),
        'Projection num: {}'.format(proj),
        'Grid spacing: {}'.format(params['grid_delta']),
        'Grid min # overlap: {}'.format(params['grid_n_min']),
        'Projection % shared variance: {:2.2f}'.format(100
                                                       * frac_shared_var)
    ]
    fh.text(
        0.05, 1 - 0.05,
        '\n'.join(title_str),
        fontsize=12,
        horizontalalignment='left',
        verticalalignment='top'
    )

    # Save figure
    fig_str = [
        'FlowComp',
        subject,
        dataset,
        params['projection_mode'],
        '{:03d}'.format(proj)
    ]
    fig_str = '_'.join(fig_str) + '.pdf'
    fig_name = os.path.join(save_dir, fig_str)
    fh.savefig(fig_name)

    # TODO: save results data here (in order to aggregate across datasets)

    # Plot results
    fh = tmp.plot_fig_4_proj_summary(results)
    fig_save_path = os.path.join(save_dir, 'Summary.pdf')
    fh.savefig(fig_save_path)
    """

    return None

