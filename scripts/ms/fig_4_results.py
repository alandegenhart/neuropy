"""Figure 4 results plotting."""

# Import
import os
import pathlib
import sys

# Define directory paths and import modules
home_dir = os.path.expanduser('~')
src_dir = os.path.join(home_dir, 'src', 'neuropy')
sys.path.append(src_dir)
import neuropy as neu
import neuropy.el.ms.fig_4 as f4


def main():
    # Arguments
    location = 'ssd'
    create_hist_plots = True

    # Define path to results
    dir_name = 'Flow10D_projMode_random_nProj_500_nPerm_100_gridDelta_2_gridNMin_2'
    _, results_path = neu.el.util.get_data_path(location)
    results_path = os.path.join(results_path, 'el_ms', 'fig_4', 'flow_10D')
    results_dir_base = os.path.join(results_path, dir_name)

    # Check to make sure directory exists
    if not os.path.isdir(results_dir_base):
        Exception('Results directory does not exist.')

    # Create directory for results
    hist_results_dir = os.path.join(results_dir_base, 'hist_results')
    os.makedirs(hist_results_dir, exist_ok=True)

    # Get iterator for items in directory
    hist_summary_data = []
    _, dirs, _ = next(os.walk(results_dir_base))
    for d in dirs:
        # Define path to pickle data and check to see if it exists
        f_name = '{}.pickle'.format(d)
        results_dir = os.path.join(results_dir_base, d)
        file_path = os.path.join(results_dir, f_name)
        if not os.path.isfile(file_path):
            continue

        # Load results data and get summary data
        results_dict = f4.load_results(file_path)
        summary_data = f4.get_flow_summary_data(results_dict)

        # Plot summary histogram for dataset
        if create_hist_plots:
            print('Plotting flow summary: {}'.format(d))
            f4.plot_flow_summary_hist(summary_data, hist_results_dir)

        # Add average normalized data to list
        hist_summary_data.append(summary_data['norm_data_mean'])

    # Plot summary across experiments
    # TODO: finish implementing this
    # TODO: verify that plotting function still returns the same results
    f4.plot_hist_summary(hist_summary_data, hist_results_dir)

    return None


if __name__ == '__main__':
    main()
