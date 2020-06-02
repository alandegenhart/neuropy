"""Figure 4 results plotting."""

# Import
import os
import sys
import fig_4_module as f4

%reload_ext autoreload
%autoreload 2

# %% Setup data paths and load data

# Define path to data
results_dir = 'Earl_20180927_projMode_random_nProj_500_nPerm_100_gridDelta_1_gridNMin_1'
results_file = '{}.pickle'.format(results_dir)
home_dir = os.path.expanduser('~')
results_dir_base = os.path.join(home_dir, 'results', 'el_ms', 'fig_4', 'flow_10D')
results_path = os.path.join(results_dir_base, results_dir, results_file)

# Load data
results_dict = f4.load_results(results_path)

# %% Run analysis

f4.plot_flow_summary_hist(results_dict)
