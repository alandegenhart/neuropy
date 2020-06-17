"""Utilities module for Energy Landscape."""

import numpy as np
import pandas as pd
import os
import re

# Custom modules
home_dir = os.path.expanduser('~')
src_dir = os.path.join(home_dir, 'src')
#sys.path.append(src_dir)
#import neuropy as neu

# Functions to implement here:
# - get paired targets
# - get colors


def data_path(location):
    """Return path to stored data."""
    location = location.lower()
    if location == 'yu':
        path = '/afs/ece.cmu.edu/project/nspg/data/batista/el/Animals'
    elif location == 'batista':
        path = ''  # Not currently set up
    elif location == 'ssd':
        path = '/Volumes/Samsung_T5/Batista/Animals'
    else:
        raise NameError('Invalid data location specified.')

    return path


class ExperimentLog:
    """Class for interacting with experiment log information."""

    def __init__(self, log='default'):
        """Class initialization method."""
        if log == 'default':
            # Specify path to default energy landscape log. The default log
            # should exist in the same directory as this module.
            self.log_path = os.path.join(
                src_dir, 'neuropy', 'el', 'EnergyLandscape_DatasetList.xlsx'
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
        for k, v in criteria:
            mask = mask & (self.log[k] == v)

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

        # Define function to get the data path for a single row
        def data_path_row(row, base_path):
            # Get list of directories and convert to file name prefix
            dir_num = range(row['dir_start'], row['dir_end'])
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

            # Get list of files in the translated directory
            _, _, translated_file_list = next(os.walk(translated_dir))

            # Iterate over dataset string prefix
            file_list = []
            for dsp in dir_str_prefix:
                # Define expression to check if file exists
                expr = re.compile(dsp)

                # Check to see if string is in any of the translated files
                for tfl in translated_file_list:
                    if expr.findall(tfl):
                        # Add to file list
                        file_list.append(os.path.splitext(tfl)[0])

            dir_list = {
                'dir_path': translated_dir,
                'file_list': file_list
            }

            return dir_list

        # Apply function to each row in the dataset
        data_path_base = data_path(location)
        ds_dir_list = self.log.apply(
            data_path_row, axis=1, args=(data_path_base,)
        )

        # Append paths to log
        if in_place:
            self.log['path'] = ds_dir_list
            return None
        else:
            return ds_dir_list

