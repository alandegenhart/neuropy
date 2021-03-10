"""AIBS package utilities module"""

import os
import pickle
import pandas as pd
import numpy as np


def load_boc():
    """Load brain observatory cache."""
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache

    # Create BOC object
    config = load_config()
    boc = BrainObservatoryCache(manifest_file=config['manifest_file'])

    return boc


def load_config() -> dict:
    """Load package configuration file."""
    import json

    # Get path to config file
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')

    # Load criteria
    with open(config_path, 'rb') as f:
        config = json.load(f)

    return config


def get_datasets(stimuli=None):
    """Get all local (downloaded) datasets."""
    import json

    # Load BOC
    boc = load_boc()

    # Load dataset criteria JSON
    criteria_path = os.path.join(
        home_dir, 'src', 'aibs', 'analysis', 'movie_fingerprint',
        'dataset_criteria.json'
    )
    with open(criteria_path, 'rb') as f:
        criteria = json.load(f)
        criteria = criteria['data']  # Returns list of dict

    exp_containers = []
    for c in criteria:
        # Get criteria for current set of criteria
        ec = boc.get_experiment_containers(
            targeted_structures=c['structure'],
            imaging_depths=c['depth'],
            cre_lines=c['cre_line']
        )
        exp_containers.extend(ec)

    # Create dataframe with all experiments to process
    exp_containers = pd.DataFrame(exp_containers)  # NOTE -- probably don't need pandas here

    # Each of the experiment containers will contain 3 different experiments.
    # Iterate over these to get valid experiments.
    exp_ids = []
    for ec in exp_containers.id:
        # For each experiment container ID, get the valid experiments
        exps = boc.get_ophys_experiments(
            experiment_container_ids=[ec],
            stimuli=[stimuli]
        )
        exps = pd.DataFrame(exps)  # Again, don't need pandas here
        exp_ids.extend(exps.id)

    return boc, exp_ids


def get_downloaded_experiments(exp_list) -> list:
    """Return subset of experiments that have been downloaded."""

    # Get location of events and data directories
    boc = load_boc()
    experiment_data_dir = boc.get_cache_path(None, 'EXPERIMENT_DATA')
    events_dir = boc.get_cache_path(None, 'EVENTS_DATA')

    # Load BOC
    boc = load_boc()

    # Iterate over experiments
    valid_experiments = []
    experiment_data_file_list = []
    events_file_list = []
    for e in exp_list:
        # Get file paths for experiment data and events
        experiment_data_file_path = experiment_data_dir % e
        events_file_path = events_dir % e

        # Check if both the experiment data and events files exist.  If so, add
        # them to the list of valid experiments
        if os.path.exists(experiment_data_file_path) and os.path.exists(events_file_path):
            valid_experiments.append(e)
            experiment_data_file_list.append(experiment_data_file_path)
            events_file_list.append(events_file_path)

    return valid_experiments, experiment_data_file_list, events_file_list


def save_results(data, base_path, file_name):
    """Results saving helper function.

    This function is essentially a wrapper around the built-in 'pickle'
    functionality in Python.  It will pickle the input data and save it in the
    specified location.  Data will be saved with the '*.results' extension,
    which is used in related load functions.
    """

    file_path = os.path.join(base_path, file_name + '.results')
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

    return None


def load_results_dir(dir_path: str, proc_func=None, args=[], kwargs=[]) -> list:
    """Load all results files from directory."""

    # Get all *.results files in the provided data path
    all_files = os.listdir(dir_path)
    valid_files = [f for f in all_files if f.endswith('.results')]

    # Iterate over results file, load, and append to the array
    data_list = []
    for vf in valid_files:
        file_path = os.path.join(dir_path, vf)
        with open(file_path, 'rb') as f:
            # Load data, and process if function is provided
            data = pickle.load(f)
            if proc_func is not None:
                data = proc_func(data, *args, **kwargs)

            data_list.append(data)

    return data_list


def dict_to_str(d, val_sep=' ', item_sep=' '):
    """Convert dict to string"""
    str_list = [f'{k}{val_sep}{v}' for k, v in d.items()]
    s = item_sep.join(str_list)
    return s


def load_metrics():
    """Load metrics data"""

    config = load_config()
    file_loc = os.path.join(config['data_dir'], 'processed', 'metrics_clean.csv')
    metrics = pd.read_csv(file_loc, low_memory=False)
    metrics.drop(columns=['Unnamed: 0'], inplace=True)
    metrics.set_index('cell_specimen_id', inplace=True)

    return metrics


def load_rf_data(cell_id_list):
    """Load saved receptive field responses."""

    # Load RF data
    config = load_config()
    file_loc = os.path.join(config['data_dir'], 'processed', 'rf_processed.pkl')
    with open(file_loc, 'rb') as f:
        rf_df = pickle.load(f)

    # Filter data. The rf dataframe can have multiple rows per neuron -- these
    # correspond to different stimuli (locally sparse noise 4deg and 8deg). In
    # general, the 8 degree stimuli apparently result in cleaner RF estimates
    # due to their larger size.

    # Get duplicate cell ids
    counts = rf_df.cell_specimen_id.value_counts()
    counts = counts[counts > 1]

    # It is possible to iterate over rows and remove them (see below). However,
    # this can take a non-trivial amount of time if there are a lot of rows to
    # filter through. Instead, find a mask for all rows with duplicate cell ids
    # and a stimulus other than lsn 8deg.
    duplicate_mask = rf_df.cell_specimen_id.isin(counts.index)
    stimulus_mask = np.logical_not(rf_df.stimulus.isin(['locally_sparse_noise_8deg']))
    valid_mask = np.logical_not(np.logical_and(duplicate_mask, stimulus_mask))
    rf_df = rf_df[valid_mask]

    """
    # Iterate over duplicate cells
    for cid in counts.index:
        # Get rows of the dataframe with the duplicate id
        mask = rf_df.cell_specimen_id.isin([cid])
        duplicate_rows = rf_df[mask]

        # Mask out unwanted stimulus
        stim_mask = duplicate_rows.stimulus.isin(['locally_sparse_noise_8deg'])
        drop_row_id = duplicate_rows.index[np.logical_not(stim_mask)]

        # Drop corresponding row
        assert len(drop_row_id) == 1, 'Unexpected number of invalid rows found.'
        rf_df.drop(index=drop_row_id[0], inplace=True)
    """

    rf_mask = rf_df.cell_specimen_id.isin(cell_id_list)
    rf_df = rf_df[rf_mask]

    # Split rf dataframe into two series. Drop all NaNs, which indicate that
    # there was not a valid RF. Also discard extra RFs (in cases where multiple
    # receptive fields were found.
    rf_df.set_index('cell_specimen_id', inplace=True)
    rf_on = rf_df.rf_on.dropna()
    rf_off = rf_df.rf_off.dropna()
    rf_on = rf_on.apply(lambda x: x[0, :])
    rf_off = rf_off.apply(lambda x: x[0, :])

    # Get on and off masks
    rf_on_mask = pd.Series(cell_id_list).isin(rf_on.index)
    rf_off_mask = pd.Series(cell_id_list).isin(rf_off.index)

    # RF information is currently only specified for neurons with valid RFs.
    # Create new series the same size as the original data
    n_neurons = len(cell_id_list)
    rf_on_all = pd.Series(data=[np.zeros(2)] * n_neurons, index=cell_id_list)
    rf_on_all.loc[rf_on.index] = rf_on
    rf_off_all = pd.Series(data=[np.zeros(2)] * n_neurons, index=cell_id_list)
    rf_off_all.loc[rf_off.index] = rf_off

    rf_info = {
        'on': rf_on_all,
        'on_mask': rf_on_mask,
        'off': rf_off_all,
        'off_mask': rf_off_mask
    }

    return rf_info