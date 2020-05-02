#%% Import

# Standard modules
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custom modules
sys.path.append('/Users/alandegenhart/src/')
import neuropy as neu
from neuropy.analysis import gpfa
import neuropy.temp as tmp

# Setup autoreload
%reload_ext autoreload
%autoreload 2

#%% Load data, decoding, and GPFA parameters

# Define data paths
base_dir = os.path.join(os.sep, 'Volumes', 'Samsung_T5', 'Batista', 'Animals')
subject = 'Earl'
dataset = '20180927'
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
int_dec_path = os.path.join(data_dir, df_int['decoderName'].iloc[0] + '.mat')
rot_dec_path = os.path.join(data_dir, df_rot['decoderName'].iloc[0] + '.mat')

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
onset_idx_int = df_int.apply(tmp.find_traj_onset, axis=1, result_type='expand')
df_int['trajOnsetIdx'] = onset_idx_int['trajOnset']
df_int['trajOffsetIdx'] = onset_idx_int['trajOffset']
onset_idx_rot = df_rot.apply(tmp.find_traj_onset, axis=1, result_type='expand')
df_rot['trajOnsetIdx'] = onset_idx_rot['trajOnset']
df_rot['trajOffsetIdx'] = onset_idx_rot['trajOffset']

# Check re-decode accuracy (intuitive)
dif = neu.el.validation.redecode_cursor_pos(
    df_int['decodeSpikeCounts'], df_int['decodeState'], dec_int
)
# Check re-decode accuracy (rotated)
dif = neu.el.validation.redecode_cursor_pos(
    df_rot['decodeSpikeCounts'], df_rot['decodeState'], dec_rot
)

# Remove non-paired targets
df_int, targ_cond_int, targ_info_int = tmp.remove_non_paired_trials(df_int)
df_rot, targ_cond_rot, targ_info_rot = tmp.remove_non_paired_trials(df_rot)

#%% Plot trajectories

col_map = tmp.define_color_map()

# Get new truncated trajectory values
traj_valid = df_int.apply(
    lambda x: x['decodeState'][:, x['trajOnsetIdx']:(x['trajOffsetIdx'] + 1)],
    axis='columns')

# Plot all trajectories
tmp.plot_traj(
    traj_valid,
    pd.Series(targ_cond_int),
    col_map,
    col_mode='light',
    line_width=0.5,
    marker_size=7)

# Plot average trajectories

# Get unique paired target conditions
uni_targ_cond_int = set(targ_cond_int)

for uc in uni_targ_cond_int:
    # Get subset of trials for a single target condition and average
    cond_mask = [True if tc == uc else False for tc in targ_cond_int]
    avg_traj = tmp.spatial_average(traj_valid[cond_mask])

    # Plot average trajectory
    tmp.plot_single_traj(avg_traj, uc, col_map, line_width=2)

# Display plot
plt.show()

#%% Test FlowField class

# Get subset of data
cond = 'T1T5'
cond_mask = [True if tc == cond else False for tc in targ_cond_int]
traj_cond = traj_valid[cond_mask]

# Define fitting parameters
delta = 10
max_dist = 200

# Fit flow field -- voxel method
F = tmp.FlowField()
F.fit(traj_cond, delta, max_dist)
F.plot(min_n=3, color=col_map[cond]['dark'])
ax_lim = np.array([-1, 1]) * 125
plt.xlim(ax_lim)
plt.ylim(ax_lim)
plt.show()

# Fit flow field -- Gaussian averaging method
F2 = tmp.GaussianFlowField()
F2.fit(traj_cond, delta, max_dist, l_const=10)
F2.plot()
plt.xlim(ax_lim)
plt.ylim(ax_lim)
plt.show()

F2.l_const

#%% Transform neural trajectories

# The idea here is to compare projections of neural activity between the
# intuitive and rotated spaces.  For each unique projection, we will want to:
# (1) visualize the trajectories and flow field for the two conditions
# (2) compare the flow fields for the two conditions
# (3) compare the comparision in (2) to that found by re-sampling.
#
# We then want to repeat (1) - (3) for many projections

# Orthonormalize
C_orth, T, s, VH = gpfa.orthogonalize(gpfa_int['C'])

# Get neural trajectories for intuitive mapping
U_int = df_int['decodeSpikeCounts'].apply(
    neu.analysis.gpfa.extract_causal_traj, args=(dec_int,)
)
# Get neural trajectories for rotated mapping -- use the intuitive mapping
# decoding parameters
U_rot = df_rot['decodeSpikeCounts'].apply(
    neu.analysis.gpfa.extract_causal_traj, args=(dec_int,)
)

# Limit trajectories to the valid portion for each trial. Might be possible to
# do this using list comprehension, but it is a bit tricky here b/c the neural
# trajectories have been removed from the dataframe

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

# Eventually we'll want to define random projections. For now, hard-code
# projections of the top 4 orthonormalized dimensions
p_1 = np.zeros((2, dec_int['xDim']))
p_2 = np.zeros((2, dec_int['xDim']))
p_1[0, 0] = 1
p_1[1, 1] = 1
p_2[0, 2] = 1
p_2[1, 3] = 1
P = np.array([p_1, p_2])

# TODO: add intuitive projection here as a sanity check

# Get unique targets to plot. Should only have to do this for the intuitive
# trials b/c both the intuitive and rotated use the same target config.
targ_cond_unique = set(targ_cond_int)

# Setup figure size
save_dir = '/Users/alandegenhart/results/el_ms/fig_4/proj/'
fig_size = (20, 10)
n_row = 2
n_col = 4

# Grid parameters
delta_grid = 1
n_grid_min = 2

# Iterate over projections
n_proj = P.shape[0]
fh = []
for proj in range(n_proj):
    # Setup figure. For now, each projection is a separate figure
    fig = plt.figure(figsize=fig_size)

    # Apply projection to data
    U_proj_int = U_orth_int.apply(lambda U: P[proj, :, :] @ U)
    U_proj_rot = U_orth_rot.apply(lambda U: P[proj, :, :] @ U)

    # Iterate over unique target conditions
    row = 0
    for tcu in targ_cond_unique:
        # Get target condition masks for the two conditions
        tc_mask_int = [True if tc == tcu else False for tc in targ_cond_int]
        tc_mask_rot = [True if tc == tcu else False for tc in targ_cond_rot]

        # Get the range of the data. We only want to fit flow fields to the region
        # where we have data points
        U_concat_int = np.concatenate(
            U_proj_int[tc_mask_int].to_numpy(), axis=1)
        U_concat_rot = np.concatenate(
            U_proj_rot[tc_mask_rot].to_numpy(), axis=1)
        U_concat = np.concatenate([U_concat_int, U_concat_rot], axis=1)
        max_U = U_concat.max(axis=1, keepdims=True)
        min_U = U_concat.min(axis=1, keepdims=True)
        lim_U = np.concatenate([min_U, max_U], axis=1)
        center = lim_U.mean(axis=1, keepdims=True)
        max_dist = (lim_U - center)[:, 1].max() + delta_grid

        # Setup axis - intuitive
        plot_idx = row * 4 + 1  # Apparently subplot indices are 1-indexed
        plt.subplot(n_row, n_col, plot_idx)

        # Fit flow field to intuitive trajectories
        F_int = tmp.FlowField()
        F_int.fit(U_proj_int[tc_mask_int], delta_grid, center, max_dist)
        F_int.plot(min_n=n_grid_min, color=col_map[tcu]['dark'])

        # Plot intuitive trajectories
        tmp.plot_traj(
            U_proj_int[tc_mask_int],
            pd.Series(targ_cond_int)[tc_mask_int],
            col_map,
            col_mode='light',
            line_width=0.5,
            marker_size=7)
        plt.title('Proj: {}, Targ: {}, Dec: {}'.format(proj, tcu, 'intuitive'))

        # Setup axis - rotated
        plot_idx = row * 4 + 2
        plt.subplot(n_row, n_col, plot_idx)

        # Fit flow field to rotated trajectories
        F_rot = tmp.FlowField()
        F_rot.fit(U_proj_rot[tc_mask_rot], delta_grid, center, max_dist)
        F_rot.plot(min_n=n_grid_min, color=col_map[tcu]['dark'])

        # Plot rotated trajectories
        tmp.plot_traj(
            U_proj_rot[tc_mask_rot],
            pd.Series(targ_cond_rot)[tc_mask_rot],
            col_map,
            col_mode='light',
            line_width=0.5,
            marker_size=7)
        plt.title('Proj: {}, Targ: {}, Dec: {}'.format(proj, tcu, 'rotated'))

        # Compare flow fields
        dX_diff = F_int.dX_fit - F_rot.dX_fit
        dX_diff_mag = np.linalg.norm(dX_diff, axis=2)  # n_grid x n_grid

        # Plot colormap
        # Setup axis - colormap
        plot_idx = row * 4 + 3  # Apparently subplot indices are 1-indexed
        axh = plt.subplot(n_row, n_col, plot_idx)

        axh.matshow(dX_diff_mag.T)
        axh.set_ylim(0, F_int.n_grid + 1)
        axh.xaxis.tick_bottom()
        axh.set_title('Difference in flow')

        # Plot histogram of pairwise-differences
        # Setup axis - histogram
        plot_idx = row * 4 + 4  # Apparently subplot indices are 1-indexed
        axh = plt.subplot(n_row, n_col, plot_idx)

        # Get histogram data and remove any NaNs
        # TODO: also filter this by overlap
        hist_data = dX_diff_mag.flatten()
        hist_data = hist_data[np.logical_not(np.isnan(hist_data))]
        axh.hist(hist_data)

        axh.set_xlabel('Flow difference magnitude')
        axh.set_ylabel('Counts')

        row += 1  # Increment row counter (TODO: find a way to remove this)

    # Save figure
    fig_name = '{}proj_{}.pdf'.format(save_dir, proj)
    plt.savefig(fig_name)

#%% Notes

# Notes [2020.04.29]
# The two methods give different results. The voxel-based approach gives a noisier
# estimate of the flow field. However, this estimate is closer to the real data.
# The weighted averaging approach gives a much smoother estimate of the flow field.
# However, the current approach is sub-optimal in that distant points can have fairly
# large flow vectors. Ideally, it would be good to update this to ignore/down-weight
# regions of the space that are sparsely-populated.

# Next steps

# Things to do soon:
# TODO: flip y-axis of decode state (currently this is in the Host coordinate system)
# TODO: trajectory plotting - int and rotated in the same projection
# TODO: new color mapping with slightly different colors for intuitive and rotated
# TODO:  Implement FlowField class
#    - NN-based method

# Analysis:
# TODO: Quantification of similarity for two flow fields
# TODO: Random projections

# Things to do eventually:
# TODO: Outer wrapper to generate plots (create figure layout, etc.)
# TODO: Add target plotting
# TODO: - `load_test.py` will need to be turned into a more standalone function
#  to batch iterate over datasets
# TODO: Look into Pandas pickle warnings in `load_test.py`

