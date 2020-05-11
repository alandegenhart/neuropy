# Setup -- Import modules

# Standard modules
import os
import sys
import numpy as np
import scipy
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

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
    U_orth_int = U_int.apply(lambda x: T @ x)
    U_orth_rot = U_rot.apply(lambda x: T @ x)

    # Get unique targets to plot. Should only have to do this for the intuitive
    # trials b/c both the intuitive and rotated use the same target config.
    targ_cond_unique = set(targ_cond_int)

    # Define colormap (used for plotting)
    col_map = tmp.define_color_map()
    line_col = {
        'int': 'xkcd:medium blue',
        'rot': 'xkcd:blood red',
        'introt': 'xkcd:gold'
    }

    # Check parameters. If the projection mode is 'orth', the maximum number of
    # projections and plotted projections is half the dimensionality
    if params['projection_mode'] == 'orth':
        n_orth_proj_max = int(x_dim/2)
        params['n_proj'] = min(params['n_proj'], n_orth_proj_max)
        params['n_proj_plots'] = min(params['n_proj_plots'], n_orth_proj_max)

    # Initialize results fields
    results = {
        'subject': subject,
        'dataset': dataset,
        'params': params,
        'targ_cond': [],
        'diff_int': [],
        'diff_rot': [],
        'diff_int_rot': [],
        'n_overlap_int': [],
        'n_overlap_rot': [],
        'n_overlap_int_rot': [],
        'proj_shared_var': []
    }

    # Create/set saving directory
    home_dir = os.path.expanduser('~')
    save_dir_base = os.path.join(home_dir,
                                 'results', 'el_ms', 'fig_4', 'flow_10D')
    params_str = [
        results['subject'],
        results['dataset'],
        'projMode_{}'.format(params['projection_mode']),
        'nProj_{}'.format(params['n_proj']),
        'nPerm_{}'.format(params['n_permute']),
        'gridDelta_{}'.format(params['grid_delta']),
        'gridNMin_{}'.format(params['grid_n_min'])
    ]
    params_str = '_'.join(params_str)
    save_dir = os.path.join(save_dir_base, params_str)
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over projections
    for proj in range(params['n_proj']):
        # Display status
        print('Projection: {} of {}'.format(proj + 1, params['n_proj']))

        # Create figure if necessary
        if proj < params['n_proj_plots']:
            fh, axh = tmp.subplot_fixed(2, 5, [300, 300])

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

        # Apply projection to data
        U_proj_int = U_orth_int.apply(lambda U: P.T @ U)
        U_proj_rot = U_orth_rot.apply(lambda U: P.T @ U)

        # Iterate over unique target conditions
        row = 0
        for row, tcu in enumerate(targ_cond_unique):
            # Get target condition masks for the two conditions
            tc_mask_int = [True if tc == tcu else False for tc in targ_cond_int]
            tc_mask_rot = [True if tc == tcu else False for tc in targ_cond_rot]

            # Get the range of the data. We only want to fit flow fields to the
            # region where we have data points
            U_concat_int = np.concatenate(
                U_proj_int[tc_mask_int].to_numpy(), axis=1)
            U_concat_rot = np.concatenate(
                U_proj_rot[tc_mask_rot].to_numpy(), axis=1)
            U_concat = np.concatenate([U_concat_int, U_concat_rot], axis=1)
            max_U = U_concat.max(axis=1, keepdims=True)
            min_U = U_concat.min(axis=1, keepdims=True)
            lim_U = np.concatenate([min_U, max_U], axis=1)
            center = lim_U.mean(axis=1, keepdims=True)
            max_dist = (lim_U - center)[:, 1].max() + params['grid_delta']

            # Get the set of valid indices for the intuitive and rotated
            # datasets. We will use these when subsampling.
            n_int = np.floor(sum(tc_mask_int)/2).astype(int)
            n_rot = np.floor(sum(tc_mask_rot)/2).astype(int)
            idx_int = np.argwhere(tc_mask_int)[:, 0]  # shape: (n_trials, )
            idx_rot = np.argwhere(tc_mask_rot)[:, 0]  # shape: (n_trials, )
            n_trials = np.min([n_int, n_rot])

            # Initialize arrays
            diff_int_all = []
            diff_rot_all = []
            diff_introt_all = []
            n_overlap_int = []
            n_overlap_rot = []
            n_overlap_introt = []

            # Iterate over permutations
            # TODO: possibly streamline this?
            rng = np.random.default_rng()
            for p in range(params['n_permute']):
                # Randomly permute indices.
                rnd_idx_int = rng.permutation(idx_int.shape[0])
                rnd_idx_rot = rng.permutation(idx_rot.shape[0])
                idx_perm_int = idx_int[rnd_idx_int]
                idx_perm_rot = idx_rot[rnd_idx_rot]

                # Split data. This is a bit inefficient, but we're doing it here
                # to make things clearer
                U_perm_int_1 = U_proj_int.iloc[idx_perm_int[0:n_trials]]
                U_perm_int_2 = U_proj_int.iloc[idx_perm_int[n_trials:n_trials*2]]
                U_perm_rot_1 = U_proj_rot.iloc[idx_perm_rot[0:n_trials]]
                U_perm_rot_2 = U_proj_rot.iloc[idx_perm_rot[n_trials:n_trials*2]]

                # Fit flow field to each dataset
                F_int_1 = tmp.FlowField()
                F_int_1.fit(U_perm_int_1, params['grid_delta'], center, max_dist)
                F_int_2 = tmp.FlowField()
                F_int_2.fit(U_perm_int_2, params['grid_delta'], center, max_dist)
                F_rot_1 = tmp.FlowField()
                F_rot_1.fit(U_perm_rot_1, params['grid_delta'], center, max_dist)
                F_rot_2 = tmp.FlowField()
                F_rot_2.fit(U_perm_rot_2, params['grid_delta'], center, max_dist)

                # Compare intuitive and rotated datasets to themselves. Also
                # compare intuitive and rotated to one another.
                diff_int = tmp.compare_flow_fields(
                    F_int_1, F_int_2, n_min=params['grid_n_min'])
                diff_rot = tmp.compare_flow_fields(
                    F_rot_1, F_rot_2, n_min=params['grid_n_min'])
                diff_introt_1 = tmp.compare_flow_fields(
                    F_int_1, F_rot_1, n_min=params['grid_n_min'])
                diff_introt_2 = tmp.compare_flow_fields(
                    F_int_2, F_rot_2, n_min=params['grid_n_min'])

                # Add results to list
                diff_int_all.append(diff_int['diff'])
                diff_rot_all.append(diff_rot['diff'])
                diff_introt_all.append(diff_introt_1['diff'])
                n_overlap_int.append(diff_int['n_overlap'])
                n_overlap_rot.append(diff_rot['n_overlap'])
                n_overlap_introt.append(diff_introt_1['n_overlap'])

            # --- Summarize performance for the current projection/target pair ---
            # Things to keep track of here:
            # 1. Some measure of flow similarity (median?)
            # 2. Degree of voxel overlap
            # 3. Any p-values from statistical tests
            # 4. Amount of shared variance for the projection

            # Run statistical tests
            diff_int_concat = np.concatenate(diff_int_all)
            diff_rot_concat = np.concatenate(diff_rot_all)
            diff_introt_concat = np.concatenate(diff_introt_all)

            stats_int_introt = stats.ranksums(diff_int_concat,
                                              diff_introt_concat)
            stats_rot_introt = stats.ranksums(diff_rot_concat,
                                              diff_introt_concat)

            # Add data for the current projection to the results
            results['targ_cond'].append(tcu)
            results['diff_int'].append(np.median(diff_int_concat))
            results['diff_rot'].append(np.median(diff_rot_concat))
            results['diff_int_rot'].append(np.median(diff_introt_concat))
            results['n_overlap_int'].append(np.mean(n_overlap_int))
            results['n_overlap_rot'].append(np.mean(n_overlap_rot))
            results['n_overlap_int_rot'].append(np.mean(n_overlap_introt))
            results['proj_shared_var'].append(frac_shared_var)

            # --- Plot results for current target condition ---

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

    return results

