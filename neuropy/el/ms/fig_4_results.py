"""Figure 4 results plotting."""

# Import
import os
import pathlib
import sys
import fig_4_module as f4

%reload_ext autoreload
%autoreload 2

# %% Setup data paths and load data

# Define path to data
results_dir = 'Earl_20180927_projMode_random_nProj_500_nPerm_100_gridDelta_1_gridNMin_1'
results_dir = 'Earl_20180927_projMode_random_nProj_500_nPerm_100_gridDelta_2_gridNMin_2'
results_file = '{}.pickle'.format(results_dir)
home_dir = os.path.expanduser('~')
results_dir_base = os.path.join(home_dir, 'results', 'el_ms', 'fig_4', 'flow_10D')
results_path = os.path.join(results_dir_base, results_dir, results_file)

# Load data
results_dict = f4.load_results(results_path)

# %% Get list of directories

# Get iterator for items in directory
dirs = os.scandir(results_dir_base)
for d in dirs:
    # Check to see if the item is a directory
    if not d.is_dir():
        continue

    # Define path to pickle data and check to see if it exists
    f_name = '{}.pickle'.format(d.name)
    file_path = os.path.join(results_dir_base, d.name, f_name)
    P = pathlib.Path(file_path)
    if not P.exists():
        continue
        
    # Load results data and analyze
    results_dict = f4.load_results(file_path)
    f4.plot_flow_summary_hist(results_dict)
    
