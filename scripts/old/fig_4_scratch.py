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


#%% Check re-decode accuracy

# Check re-decode accuracy (intuitive)
dif = neu.el.validation.redecode_cursor_pos(
    df_int['decodeSpikeCounts'], df_int['decodeState'], dec_int
)
# Check re-decode accuracy (rotated)
dif = neu.el.validation.redecode_cursor_pos(
    df_rot['decodeSpikeCounts'], df_rot['decodeState'], dec_rot
)

