"""Utilities module for Energy Landscape."""

import numpy as np
import pandas as pd
import os
import re

# Get directory paths
home_dir = os.path.expanduser('~')
src_dir = os.path.join(home_dir, 'src')

# Functions to implement here:
# - get paired targets
# - get colors


def get_valid_criteria():
    """Return criteria defining valid datasets."""
    # TODO: perhaps this should be analysis-specific?
    criteria = {
        'subject': 'Earl',
        'grid': 'targ_4',
        'int_calib_method': 'center_out_udp',
        'bins': 7,
        'align_bins': 1,
        'align': 'start',
        'gpfa_int': 'grad_train',
        'gpfa_rot': 'cond_4'
    }
    return criteria


def get_data_path(location):
    """Return path to stored data."""
    location = location.lower()
    if location == 'yu':
        data_loc = '/afs/ece.cmu.edu/project/nspg/data/batista/el/Animals'
        results_loc = '/afs/ece.cmu.edu/project/nspg/adegenha/results'
    elif location == 'batista':
        data_loc = ''  # Not currently set up
        results_loc = ''
    elif location == 'ssd':
        data_loc = '/Volumes/Samsung_T5/Batista/Animals'
        results_loc = '/Users/alandegenhart/results'
    else:
        raise NameError('Invalid data location specified.')

    return data_loc, results_loc


class ExperimentLog:
    """Class for interacting with experiment log information."""

    def __init__(self, log='default'):
        """Class initialization method."""
        if log == 'default':
            # Specify path to default energy landscape log. The default log
            # should exist in the same directory as this module.
            self.log_path = os.path.join(
                src_dir, 'neuropy', 'neuropy', 'el',
                'EnergyLandscape_DatasetList.xlsx'
            )
        else:
            self.log_path = ''

        # Initialize other attributes here
        self.log = None

    def load(self, log_path=None):
        """Load log file from Excel spreadsheet."""

        # Add log path if specified. This can be added here or in the init call
        if log_path is not None:
            self.log_path = log_path

        # Load spreadsheet into Pandas object
        self.log = pd.read_excel(self.log_path)

    def apply_criteria(self, criteria, in_place=True):
        """Filter datasets by applying criteria.

        This function applies the criteria specified in criteria to the set of
        datasets contained in the log. Criteria should be specified as a dict
        of key/value pairs corresponding to log column names/values.
        """

        # Iterate over keys in criteria dict
        mask = pd.Series(np.full(self.log.shape[0], True))
        for k, v in criteria.items():
            mask = mask & (self.log[k] == v)
            # TODO: if multiple values are specified, check for each value

        # Apply filter to data. If the in_place flag is set to true, update the
        # data in the object, otherwise return the data frame with the filter
        # applied.
        if in_place:
            self.log = self.log[mask]
            return None
        else:
            return self.log[mask]

    def get_data_path(self, location, in_place=True):
        """Get data paths for datasets."""

        def data_path_row(row, base_path):
            """Return path to data and a list of data files for a single entry
            in the log DataFrame.
            """

            # Get list of directories and convert to file name prefix
            dir_num = range(row['dir_start'], row['dir_end'] + 1)
            dir_str_prefix = [
                '{}{}_{:02d}_'.format(row['subject'], row['dataset'], dn)
                for dn in dir_num
            ]

            # Define path to data
            ds_str = str(row['dataset'])
            yr = ds_str[0:4]
            mo = ds_str[4:6]
            translated_dir = os.path.join(
                    base_path, row['subject'], yr, mo, ds_str, 'translated'
            )

            # Get list of files in the translated directory.
            if os.path.isdir(translated_dir):
                _, _, translated_dir_file_list = next(os.walk(translated_dir))
            else:
                # The desired directory didn't exist. Return None.
                return None, None

            # Filter out all files except those that end in _SI_translated.mat
            expr = re.compile('_SI_translated.mat')
            translated_file_list = [
                tfl for tfl in translated_dir_file_list if expr.findall(tfl)
            ]

            # Note -- in some cases (particularly when running this locally on
            # the data stored on the SSD), the 'SI_translated' files appear to
            # have been re-named. As a fall-back, grab all files that end in
            # '_SI.mat' if there are no files that end in '_SI_translated.mat'.
            if translated_file_list == []:
                expr = re.compile('_SI.mat')
                translated_file_list = [
                    tfl for tfl in translated_dir_file_list if expr.findall(tfl)
                ]

            # Iterate over dataset string prefix
            file_list = []
            for dsp in dir_str_prefix:
                # Define expression to check if file exists. To do this, look
                # for files in the translated
                expr = re.compile(dsp)

                # Check to see if string is in any of the translated files
                for tfl in translated_file_list:
                    if expr.findall(tfl):
                        # Remove 'translated' from the dir list if it exists
                        file_name = os.path.splitext(tfl)[0]
                        trans_str_idx = file_name.find('_translated')
                        if trans_str_idx != -1:
                            file_name = file_name[0:trans_str_idx]

                        # Add to file list
                        file_list.append(file_name)

            return translated_dir, file_list

        # Apply function to each row in the dataset
        # TODO: Update to use the apply() method. This should be more efficient
        data_path_base, _ = get_data_path(location)
        #ds_dir_list = self.log.apply(
        #    data_path_row, axis=1, args=(data_path_base,)
        #)

        # Iterate over rows -- this is inefficient, but is done for debugging
        # purposes
        ds_dir_path = []
        ds_dir_list = []
        for row in self.log.iterrows():
            row_dir, row_files = data_path_row(row[1], data_path_base)
            ds_dir_path.append(row_dir)
            ds_dir_list.append(row_files)

        # Append paths to log
        if in_place:
            self.log['dir_path'] = ds_dir_path
            self.log['dir_list'] = ds_dir_list
            return None
        else:
            return ds_dir_path, ds_dir_list

    def get_experiment_sets(self, task_list):
        """Get set of files for an experiment.

        This function returns a DataFrame where each row is a single experiment
        consisting of a set of tasks.
        """

        # Find unique subject/dataset/condition combinations
        unique_cols = ['subject', 'dataset', 'condition']
        unique_datasets = self.log.drop_duplicates(unique_cols)[unique_cols]

        # Initialize dict to hold experiments
        experiment_sets = {
            'subject': [],
            'dataset': [],
            'condition': [],
            'dir_path': []
        }
        # Add fields for tasks
        for task in task_list:
            experiment_sets[task] = []

        # Iterate over unique datasets and get files
        for uds in unique_datasets.iterrows():
            # Get entries in the full log for the current dataset
            uds = uds[1]  # Don't care about the index

            # Generate unique experiment mask. Explicitly generate masks for
            # the unique columns.
            subject_mask = self.log['subject'] == uds['subject']
            dataset_mask = self.log['dataset'] == uds['dataset']
            if uds['condition'] is np.nan:
                # Note: nan != nan, so don't u
                condition_mask = self.log['condition'].isna()
            else:
                condition_mask = self.log['condition'] == uds['condition']

            mask = subject_mask & dataset_mask & condition_mask
            dataset = self.log[mask]

            # Iterate over tasks. If the task exists, get the directory list.
            # Otherwise, set the corresponding value to None
            for task in task_list:
                # Get task directory list
                task_mask = dataset['task'] == task
                task_dir_list = list(dataset['dir_list'][task_mask].array)
                if (len(task_dir_list) == 0) or task_dir_list == [[]]\
                        or task_dir_list == [None]:
                    task_dir_list = None

                # Add to output dict
                experiment_sets[task].append(task_dir_list)

            # Add other information to the output dict
            for uc in unique_cols:
                experiment_sets[uc].append(uds[uc])

            # Add the first 'dir_path' entry, as these should all be the same
            # for a given dataset
            experiment_sets['dir_path'].append(dataset['dir_path'].iloc[0])

        # Convert experiment set dict to a pandas array and remove any columns
        # where one of the two task directory lists is nan
        experiment_sets = pd.DataFrame(experiment_sets)
        mask = pd.Series([True] * experiment_sets.shape[0])
        for task in task_list:
            mask = mask & experiment_sets[task].notna()
        experiment_sets = experiment_sets[mask]

        return experiment_sets

