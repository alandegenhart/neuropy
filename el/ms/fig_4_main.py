"""Figure 4 main analysis

"""


# Import libraries
import os
import sys
import fig_4_module as f4
import multiprocessing as mp


def main_fun(args):
    """Main analysis function for a single parameter set."""
    results, flow_ex, comp_cond = f4.flow_analysis(
        args['subject'], args['dataset'], args['params'], args['dir'])
    f4.plot_flow_ex(
        args['subject'], args['dataset'], flow_ex, comp_cond, args['params'])
    f4.plot_flow_results(
        args['subject'], args['dataset'], results, args['params'])


# Check input arguments. Currently this script accepts one input argument, which
# specifies whether to use the local or remote path.
if len(sys.argv) == 1:
    mode = 'local'
else:
    mode = sys.argv[1]

# Define data path
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
use_multiproc = True
pool_size = 8

# Define parameter dictionary
params = {
    'projection_mode': ['random', 'orth'],
    'n_proj': [500],
    'n_permute': [100],
    'grid_delta': [1, 1.5, 2.5, 3],
    'grid_n_min': [1, 2, 3],
    'n_proj_plots': [20]
}

# Define possible parameter sets. Create an array of dicts to pass to the main
# function in order to use multiprocessing if desired.
param_sets = f4.define_param_sets(params)
all_params = [
    {
        'subject': subject,
        'dataset': dataset,
        'dir': base_dir,
        'params': p_set
    }
    for p_set in param_sets
]

if use_multiproc:
    with mp.Pool(processes=pool_size) as pool:
        pool.map(main_fun, all_params)
else:
    for params in all_params:
        main_fun(params)


