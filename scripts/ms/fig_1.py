"""Figure 1 script.

This script simulates a low-dimensional network embedded in a high-dimensional
space.

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
    # Setup save path
    save_dir = os.path.join(
        os.path.expanduser('~'), 'results', 'el_ms', 'fig_1_sim'
    )
    os.makedirs(save_dir, exist_ok=True)

    # Define model parameters
    model_params = define_model_params()
    P = define_high_d_projection()
    
    # Generate trajectories for example
    # Start position 1
    x_0 = np.array([[4, 0]]).T
    n_ts = 1500
    X_ab = simulate_trajectory(
        model_params, x_0,
        n_ts=n_ts
    )
    # Start position 2
    x_0 = np.array([[-3, -0]]).T
    X_ba = simulate_trajectory(
        model_params, x_0,
        n_ts=n_ts
    )
    # Convert to pd series and 3D projection
    traj_2d_ex = pd.Series([X_ab, X_ba])
    traj_3d_ex = traj_2d_ex.apply(lambda u: P @ u)
    cond = pd.Series(['T1T5', 'T5T1'])
    
    # Generate trajectories for prediction 1 (A->B and B->A paths)
    # Simulate trajectories (A->B)
    x_0 = np.array([[4, 0]]).T
    targ_pos = np.array([[-4, 0]]).T
    targ_rad = 0.1
    X_ab = simulate_trajectory(
        model_params, x_0,
        targ_pos=targ_pos,
        targ_rad=targ_rad
    )
    # Simulate trajectories (B->A)
    x_0 = np.array([[-4, 0]]).T
    targ_pos = np.array([[4, 0]]).T
    targ_rad = 0.1
    X_ba = simulate_trajectory(
        model_params, x_0,
        targ_pos=targ_pos,
        targ_rad=targ_rad
    )
    # Convert to pd series and 3D projection
    traj_2d_pred_1 = pd.Series([X_ab, X_ba])
    traj_3d_pred_1 = traj_2d_pred_1.apply(lambda u: P @ u)  # Not used
    
    # Calculate flow field
    x_lim = np.array([-5, 5])
    n_x = 8
    X, dX = calc_flow_field(model_params, x_lim, n_x)
    flow_info = {
        'X': X,
        'dX': dX
    }

    # Define manifold points
    x_lim_patch = np.array([-4.5, 4.5])
    X_plane_2d = np.array([
        [x_lim_patch[0], x_lim_patch[0]],
        [x_lim_patch[0], x_lim_patch[1]],
        [x_lim_patch[1], x_lim_patch[1]],
        [x_lim_patch[1], x_lim_patch[0]],
        [x_lim_patch[0], x_lim_patch[0]]
    ]).T
    X_plane_3d = P @ X_plane_2d

    # Plot timecourses
    fh_ts = plot_time_series(traj_3d_ex, cond)
    fh_ts.savefig(os.path.join(save_dir, 'Fig_1_timeseries.pdf'))

    # Plot 3D trajectories + manifold
    fh_3d = plot_3d_trajectories(traj_3d_ex, X_plane_3d, cond)
    fh_3d.savefig(os.path.join(save_dir, 'Fig_1_3d_traj.pdf'))

    # Plot 2D projections
    # Here, we plot 2 different 2D plots:
    # 1. A 2d representation of the 3d example
    # 2. Prediction 1 (asymmetries)

    fh_2d, axh = tmp.subplot_fixed(
        1, 2, [300, 300],
        x_margin=[150, 150],
        y_margin=[150, 200]
    )
    plot_2d_trajectories(traj_2d_ex, cond, flow_info, x_lim_patch, axh[0][0])
    plot_2d_trajectories(traj_2d_pred_1, cond, flow_info, x_lim_patch, axh[0][1])
    fh_2d.savefig(os.path.join(save_dir, 'Fig_1_2d_traj.pdf'))
    # Plot hypotheses (different initial conditions)

    return None


def define_model_params():
    """Define linear dynamics model parameters.

    Define parameters for the model:
    dx = A x + u
    """
    # Define parameters
    A = np.array([
        [0, -0.2],
        [0.1, 0]
    ])
    u = np.array([[0, 0]]).T
    tau = 50

    # Incorporate time constant into matrices
    A = A / tau
    u = u / tau

    # Add model parameters to a dict
    model_params = {
        'A': A,
        'u': u
    }

    return model_params


def simulate_trajectory(model_params, x_0,
                        n_ts=1500,
                        targ_pos=None,
                        targ_rad=None):
    """Run dynamics simulation.

    This function generates trajectories from a linear dynamics model of the
    form:

    dx = A @ x + u

    A virtual 'target' can also be added as an input.  If added, this stops the
    trajectory once it reaches the target region.
    """
    # Check target position
    if targ_pos is not None and targ_rad is not None:
        # Both target position and radius must be specified in order to do a
        # target check.
        targ_flag = True
    else:
        targ_flag = False

    # Initialize
    X = [x_0]
    x = x_0

    # Iterate over time steps
    for t in range(n_ts):
        # Calculate velocity
        dx = model_params['A'] @ x + model_params['u']
        x = x + dx
        X.append(x)

        # Perform target check (if desired)
        if targ_flag:
            d = np.linalg.norm(x - targ_pos)
            if d < targ_rad:
                break

    # Convert position list to a matrix
    X = np.concatenate(X, axis=1)  # x_dim x time

    return X


def calc_flow_field(model_params, x_lim, n_x):
    """Calculate flow field vectors for a given set of model parameters."""
    # Calculate matrices of inputs and states
    x = np.linspace(x_lim[0], x_lim[1], num=n_x)
    X_1, X_2 = np.meshgrid(x, x)
    x_1 = np.reshape(X_1, -1)
    x_2 = np.reshape(X_2, -1)
    X = np.stack([x_1, x_2], axis=0)  # x_dim x samp
    U = np.tile(model_params['u'], X.shape[1])  # x_dim x samp

    # Calculate associated flow vectors
    dX = model_params['A'] @ X + U

    return X, dX


def define_high_d_projection():
    """Define a projection from a low-d (2D) space to a high-d (3D) one.

    To define the projection, we start by defining a single axis in 3D space.
    We then define a second axis, and project it into the null space of the
    first one.  This gives us a set of orthogonal vectors that can be used to
    map 2D activity into a 3D space.
    """
    # Define the first axis
    p_1 = np.array([[1, -0.1, 0.1]]).T
    p_1 = p_1 / np.linalg.norm(p_1)  # Make a unit vector

    # Define the second axis and project into the null space of p_1
    p_2 = np.array([[0, 1, 0.6]]).T
    p_null = sp.linalg.null_space(p_1.T)
    p_2 = p_null @ p_null.T @ p_2
    p_2 = p_2 / np.linalg.norm(p_2)

    # Define projection matrix y = P @ x
    P = np.concatenate([p_1, p_2], axis=1)

    return P


def plot_time_series(X, cond):
    """Plot high-d states versus time."""
    fh, axh = tmp.subplot_fixed(
        1, 1, [300, 300],
        x_margin=[150, 150],
        y_margin=[150, 200]
    )
    curr_ax = axh[0][0]

    # Get color map and define scale factor
    col_map = tmp.define_color_map()
    sf = 0.1

    # Iterate over conditions
    for x, c in zip(X, cond):
        # Iterate over rows
        for row in range(x.shape[0]):
            x_plot = x[row, :] * sf + row
            curr_ax.plot(x_plot, color=col_map[c]['dark'])

    # Set tick labels
    curr_ax.set_yticks([0, 1, 2])
    curr_ax.set_yticklabels(['N1', 'N2', 'N3'])
    curr_ax.set_xlabel('Time')
    curr_ax.set_xticks([])

    return fh


def plot_3d_trajectories(X, X_plane, cond):
    """Plot 3-dimensional trajectories."""
    # Create axes and get color map
    fh, axh = tmp.create_3d_axes()
    col_map = tmp.define_color_map()

    # Plot manifold
    X_surf = np.array([
        X_plane[0, 0:2],
        X_plane[0, [3, 2]]
    ])
    Y_surf = np.array([
        X_plane[1, 0:2],
        X_plane[1, [3, 2]]
    ])
    Z_surf = np.array([
        X_plane[2, 0:2],
        X_plane[2, [3, 2]]
    ])
    axh.plot_surface(
        X_surf, Y_surf, Z_surf,
        color='#bf9430',
        shade=False
    )

    # Plot trajectories
    tmp.plot_traj(X, cond, col_map, axh=axh, mode='3d', line_width=2)

    # Format plot
    ax_lim = [-4, 4]
    ax_tick = [ax_lim[0], 0, ax_lim[1]]
    axh.set_xlim(ax_lim)
    axh.set_xticks(ax_tick)
    axh.set_xlabel('N1')
    axh.set_ylim([4, -4])
    axh.set_yticks([4, 0, -4])
    axh.set_ylabel('N2')
    axh.set_zlim(ax_lim)
    axh.set_zticks(ax_tick)
    axh.set_zlabel('N3')
    axh.view_init(elev=30, azim=45)

    return fh


def plot_2d_trajectories(traj, cond, flow, ax_lim, axh):
    """Plot 2-dimensional trajectories and flow field."""
    import matplotlib.patches as mpatches

    # Plot manifold
    rect_pos = (ax_lim[0], ax_lim[0])
    rect_wh = ax_lim[1] - ax_lim[0]
    rect = mpatches.Rectangle(rect_pos, rect_wh, rect_wh,
                              facecolor='#bf9430', zorder=0)
    axh.add_patch(rect)

    # Plot flow field
    axh.quiver(
        flow['X'][0, :], flow['X'][1, :], flow['dX'][0, :], flow['dX'][1, :],
        color='k'
    )

    # Plot trajectories
    col_map = tmp.define_color_map()
    tmp.plot_traj(traj, cond, col_map, line_width=2, axh=axh)

    # Format axes
    axh.set_xlim(ax_lim)
    axh.set_ylim(ax_lim)
    axh.set_xticks([])
    axh.set_yticks([])
    axh.set_xlabel(r'$\mathrm{u}_1$')
    axh.set_ylabel(r'$\mathrm{u}_2$')

    return None


if __name__ == '__main__':
    main()
