"""Figure 4 main analysis

"""


# Import libraries
import os
import sys
import multiprocessing as mp
import argparse

# Define directory paths and import modules
home_dir = os.path.expanduser('~')
src_dir = os.path.join(home_dir, 'src', 'neuropy')
sys.path.append(src_dir)
import neuropy as neu
import neuropy.el.ms.fig_4 as f4


def main():
    """Main analysis function."""
    # Parse optional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--location',
        help='Specify data location.'
    )
    parser.add_argument(
        '--dry_run',
        help='Specify if the script should actually copy/delete files.',
        action='store_true'
    )
    parser.add_argument(
        '--use_multiproc',
        help='Specify if parallel processing should be used.',
        action='store_true'
    )
    parser.add_argument('--pool_size', default=4, type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # Print arguments
    print('Arguments:')
    print('Dry-run: {}'.format(args.dry_run))
    print('Location: {}'.format(args.location))
    print('Multiproc: {}'.format(args.use_multiproc))
    print('Pool size: {}'.format(args.pool_size))
    print('Debug: {}'.format(args.debug))

    # Get valid datasets
    EL = neu.el.util.ExperimentLog()
    EL.load()
    EL.get_data_path(args.location)
    criteria = neu.el.util.get_valid_criteria()
    EL.apply_criteria(criteria)

    # Get experiment sets
    task_list = ['tt_int', 'tt_rot']
    experiment_list = EL.get_experiment_sets(task_list)
    # Define data and results locations
    data_path, results_path = neu.el.util.get_data_path(args.location)

    # Define parameter dictionary
    if args.debug:
        params = {
            'projection_mode': ['orth', 'random'],
            'n_proj': [10],
            'n_permute': [10],
            'grid_delta': [2],
            'grid_n_min': [2],
            'n_proj_plots': [20]
        }
    else:
        params = {
            'projection_mode': ['orth', 'random'],
            'n_proj': [500],
            'n_permute': [100],
            'grid_delta': [2],
            'grid_n_min': [2],
            'n_proj_plots': [20]
        }

    # Define possible parameter sets. To allow this to run over multiple
    # parameter sets and data sets, we create one long list of parameters,
    # which specifies both the dataset and the analysis parameters. This is then
    # passed to the main analysis function to run the analysis.
    param_sets = f4.define_param_sets(params)

    # Iterate over parameter sets and create results directories
    updated_param_sets = []
    for p_set in param_sets:
        # Define results location and create directory if needed
        results_dir = os.path.join(
            results_path, 'el_ms', 'fig_4', 'flow_10D')
        params_str = [
            'Flow10D',
            'projMode_{}'.format(p_set['projection_mode']),
            'nProj_{}'.format(p_set['n_proj']),
            'nPerm_{}'.format(p_set['n_permute']),
            'gridDelta_{}'.format(p_set['grid_delta']),
            'gridNMin_{}'.format(p_set['grid_n_min'])
        ]
        params_dir_name = '_'.join(params_str)
        params_dir_path = os.path.join(results_dir, params_dir_name)
        if not os.path.isdir(params_dir_path):
            print('Creating directory: {}'.format(params_dir_path))
            if not args.dry_run:
                os.makedirs(params_dir_path, exist_ok=True)

        # Add directory to parameters and update parameter set list
        p_set['params_dir'] = params_dir_path
        updated_param_sets.append(p_set)

    all_params = []
    for idx, row in experiment_list.iterrows():
        for p_set in updated_param_sets:
            # Get paths
            dataset_dir, _ = os.path.split(row['dir_path'])
            # Add parameters to dict
            param_dict = {
                'subject': row['subject'],
                'dataset': str(row['dataset']),
                'tt_int': row['tt_int'][0][0] + '_pandasData.hdf',
                'tt_rot': row['tt_rot'][0][0] + '_pandasData.hdf',
                'data_dir': dataset_dir,
                'params': p_set,
                'dry_run': args.dry_run
            }
            all_params.append(param_dict)

    if args.use_multiproc:
        with mp.Pool(processes=args.pool_size) as pool:
            pool.map(main_fun, all_params)
    else:
        for params in all_params:
            _ = main_fun(params)

    return None


def main_fun(analysis_dict):
    """Main analysis function for a single parameter set."""

    # Unpack parameters -- a bit inefficient, but makes the below code easier
    # to read/follow
    subject = analysis_dict['subject']
    dataset = analysis_dict['dataset']
    params = analysis_dict['params']
    data_dir = analysis_dict['data_dir']
    tt_int = analysis_dict['tt_int']
    tt_rot = analysis_dict['tt_rot']
    dry_run = analysis_dict['dry_run']

    # Create directory to save results and get path
    results_dir_path, results_dir_name = f4.get_results_dir(
        subject, dataset, params, params['params_dir'],
        create_dir=(not dry_run)
    )
    # Print status message
    print('Running analysis: {}'.format(results_dir_name))

    # If running in dry-run mode, return now without running the analysis
    if dry_run:
        return None

    # Run flow analysis, create results plots
    try:
        results, flow_ex, comp_cond = f4.flow_analysis(
            params, data_dir, tt_int, tt_rot
        )
        f4.plot_flow_ex(
            subject, dataset, flow_ex, comp_cond, params, results_dir_path
        )
        f4.plot_flow_results(
            subject, dataset, results, params, results_dir_path
        )
        # Pack results up into a single dict and save
        results_dict = {
            'subject': subject,
            'dataset': dataset,
            'params': params,
            'proj_results': results,
            'flow_ex': flow_ex,
            'cond': comp_cond
        }
        f4.save_results(results_dict, results_dir_name, results_dir_path)
    except:
        print('Error: {}'.format(results_dir_name))
        results = None

    return results


if __name__ == '__main__':
    main()
