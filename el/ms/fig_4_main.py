# Import libraries
import os
import fig_4_module as f4

# Define data path
mode = 'local'
if mode == 'local':
    base_dir = os.path.join(
        os.sep, 'Volumes', 'Samsung_T5', 'Batista', 'Animals'
    )
elif mode == 'remote':
    base_dir = os.path.join(
        os.sep, 'afs', 'ece.cmu.edu', 'project', 'nspg', 'data',
        'batista', 'el', 'Animals'
    )

# Define subject and dataset
subject = 'Earl'
dataset = '20180927'

# Define parameter dictionary
params = {
    'projection_mode': ['orth', 'random'],
    'n_proj': [10],
    'n_permute': [25],
    'grid_delta': [1, 2],
    'grid_n_min': [2],
    'n_proj_plots': [5]
}

# Define possible parameter sets and run analysis
param_sets = f4.define_param_sets(params)
for p_set in param_sets:
    f4.flow_analysis(subject, dataset, p_set, base_dir)

