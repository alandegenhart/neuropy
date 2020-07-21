"""Figure 4 main analysis

"""


# Import libraries
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import multiprocessing as mp
import argparse

# Define directory paths and import modules
home_dir = os.path.expanduser('~')
src_dir = os.path.join(home_dir, 'src', 'neuropy')
sys.path.append(src_dir)
import neuropy as neu
from neuropy.analysis import gpfa
import neuropy.temp as tmp
import neuropy.el.ms.fig_4 as f4


def main():
    # Parameters
    location = 'ssd'
    subject = 'Earl'
    dataset = 20190312  # Only two-target intuitive this day
    criteria_set = 'two_target_int'
    dec_num = 5  # Intuitive decoder number -- linked to GPFA results

    # Get valid datasets
    EL = neu.el.util.ExperimentLog()
    EL.load()
    EL.get_data_path(location)
    criteria = neu.el.util.get_valid_criteria(criteria_set)
    criteria['subject'] = subject
    criteria['dataset'] = dataset
    EL.apply_criteria(criteria)

    # Get experiment sets
    task_list = ['tt_int']
    experiment_list = EL.get_experiment_sets(task_list)
    # Define data and results locations
    data_path, results_path = neu.el.util.get_data_path(location)

    # TODO: Load data for all target pairs here

    # Load data
    data_dir, _ = os.path.split(experiment_list.iloc[0]['dir_path'])
    pandas_file = experiment_list.iloc[0]['tt_int'][0][0] + '_pandasData.hdf'

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
        'run{:03d}'.format(dec_num),
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

    """Step 2: Extract neural trajectories."""
    # Orthonormalize
    C_orth, T, s, VH = gpfa.orthogonalize(gpfa_params['C'])
    x_dim = T.shape[0]

    # Get neural trajectories for intuitive mapping
    U = df['decodeSpikeCounts'].apply(
        neu.analysis.gpfa.extract_causal_traj, args=(dec,)
    )

    # Limit trajectories to the valid portion for each trial. Might be possible
    # to do this using list comprehension, but it is a bit tricky here b/c the
    # neural trajectories have been removed from the dataframe

    # Truncate intuitive trajectories
    for i in range(U.shape[0]):
        idx = range(df['trajOnsetIdx'].iloc[i],
                    df['trajOffsetIdx'].iloc[i] + 1)
        U.iloc[i] = U.iloc[i][:, idx]

    # Transform to orthonormalized latents
    U_orth = U.apply(lambda u: T @ u)

    # Get unique targets to plot. Should only have to do this for the intuitive
    # trials b/c both the intuitive and rotated use the same target config.
    targ_cond_unique = set(targ_cond)

    # Get features used in identifying projections
    asym_features = neu.el.comp.get_asymmetry_features(U_orth, targ_cond)

    # TODO: Implement LDA
    # TODO: Calculate orthonormal basis for intuitive mapping
    # TODO: Generate plots to validate decomposition


if __name__ == "__main__":
    main()
