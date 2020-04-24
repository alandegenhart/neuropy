"""Temporary module for NeuroPy package.

This module is a temporary home for functions in development. As things start
to mature they will be moved to a more appropriate home. In the meantime, this
allows for more rapid iteration/integration of prototpyed functions.

"""


# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_targ_cond(pos):
    """Map target position to unique condition.
    
    Arguments:
    pos -- Array of positions
    
    Returns:
    cond -- List of target condition strings
    
    """
    
    # Define target angles. Need to do this b/c the target positions might be
    # scaled.
    targ_map = {
        -0: 'T1',
        0: 'T1',
        45: 'T2',
        90: 'T3',
        135: 'T4',
        180: 'T5',
        -180: 'T5',
        -135: 'T6',
        -90: 'T7',
        -45: 'T8'
    }
    
    # Iterate over positions, convert to angles, and get associated condition
    # strings
    ang = [np.rad2deg(np.arctan2(p[0, 1], p[0, 0])).astype(int) for p in pos]
    cond = [targ_map.get(k, 'NF') for k in ang]
    
    return cond


def get_targ_pairs(start_cond, end_cond):
    """Return boolean mask for paired target conditions."""
    
    # Define target pair and flipped target pair conditions
    cond = [''.join([s, e]) for s, e in zip(start_cond, end_cond)]
    filp_cond = [''.join([e, s]) for s, e in zip(start_cond, end_cond)]
    
    # If the flipped version of a target condition appears in the set of unique
    # target conditions, then the target has a matching condition
    uni_cond = set(cond)
    mask = [True if c in uni_cond else False for c in filp_cond]
    
    return mask


def define_color_map():
    """Define map from target condition to color.
    
    Returns a dict containing, where each element is a color for a particular
    target condition.
    """
    
    map = {
        'T1T5': {'dark': '#cc2500', 'light': '#cc9a8f'},
        'T5T1': {'dark': '#0080b3', 'light': '#8fbbcc'},
        'T2T6': {'dark': '#06a600', 'light': '#91cc8f'},
        'T6T2': {'dark': '#9b00d9', 'light': '#c698d9'},
        'T3T7': {'dark': '#c80', 'light': '#ccb88f'},
        'T7T3': {'dark': '#04c', 'light': '#8fa3cc'},
        'T4T8': {'dark': '#7c0', 'light': '#b3cc8f'},
        'T8T4': {'dark': '#c06', 'light': '#cc8fad'}
    }
    
    return map


def plot_traj(traj, cond, col_map,
              col_mode='dark',
              line_width=1,
              marker_size=7):
    """Plot trajectories
    
    Plot all trajectories for a set of trials.
    
    Arguments:
    traj -- Pandas series of trajectories (each element is a trial)
    cond -- Pandas series of condition strings
    col_map -- Dict mapping conditions to color values
    
    Keyword arguments:
    col_mode -- 
    line_width --
    marker_size --
    
    """
    # Note -- it might be possible to do this using the Pandas apply() method
    # over the rows in the data frame. However, I was not able to figure out
    # a simple way to pass optional keyword arguments to the anonymous
    # function used with this approach.
    
    # For now, use a for loop:
    for i in range(traj.shape[0]):
        plot_single_traj(traj.iloc[i], cond.iloc[i], col_map,
                         col_mode=col_mode,
                         line_width=line_width,
                         marker_size=marker_size)
    
    return None
    

def plot_single_traj(traj, cond, color_map,
                     col_mode='dark',
                     line_width=1,
                     marker_size=10):
    """Plot a single trajectory"""
    
    # Plot trajectory
    plt.plot(
        traj[0, :], traj[1, :], 
        color=color_map[cond][kwargs['col_mode']],
        linewidth=kwargs['line_width']             
    )
    # Plot start point
    plt.plot(
        traj[0, 0], traj[1, 0],
        color=color_map[cond][kwargs['col_mode']],
        marker='.',
        markersize=kwargs['marker_size'],
        markeredgecolor=None,
        markerfacecolor=color_map[cond][kwargs['col_mode']]
    )
    # Plot end point
    plt.plot(
        traj[0, -1], traj[1, -1],
        color=color_map[cond][kwargs['col_mode']],
        marker='o',
        markersize=kwargs['marker_size'],
        markeredgecolor='k',
        markerfacecolor=color_map[cond][kwargs['col_mode']]
    )
    
    return None