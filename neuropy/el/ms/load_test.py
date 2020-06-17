"""Test loading of MATLAB files in Python."""

# %% Import modules

import os
import sys
import pandas as pd

# Setup paths and import custom modules
sys.path.append('/Users/alandegenhart/src/')
#import plottools as pt
import neuropy as neu

# Setup autoreload
#%reload_ext autoreload
#%autoreload 2

# %% Define path to datasets

# TODO: Update this to generate data paths automatically. This could probably
# be done by porting over some of the functionality from the MATLAB database
# code.

base_dir = os.path.join(os.sep, 'Volumes', 'Samsung_T5', 'Batista', 'Animals')
subject = 'Earl'
dataset = '20180927'
data_dir = os.path.join(
    base_dir, subject, dataset[0:4], dataset[4:6], dataset, 'translated'
)
export_dir = os.path.join(data_dir, 'exportData')
save_dir = os.path.join(data_dir, 'pandasData')
dataset_name = [
    'Earl20180927_04_condGridTask_01_SI_exportData.mat',
    'Earl20180927_05_twoTargetABBA_rotated_01_SI_exportData.mat'
]
os.makedirs(save_dir, exist_ok=True)

# %% Load data and convert to Pandas DataFrame

# Load MATLAB data and convert data to dict/list format
for file in dataset_name:
    print('Converting {} ... '.format(file), end='')
    # Load data, convert to DataFrame, clean
    data = neu.util.convertmat.convert_mat(os.path.join(export_dir, file))
    df = pd.DataFrame(data['S'])
    neu.el.proc.clean_trajectory_data(df)

    # Save
    df.to_hdf(
        os.path.join(save_dir, os.path.splitext(file)[0] + '.hdf'), key='df'
    )
    print('done.')

