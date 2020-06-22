"""Convert MATLAB export data to pandas format."""

# Import standard modules
import os
import sys
import argparse
import pandas as pd
import multiprocessing as mp

# Custom modules
home_dir = os.path.expanduser('~')
neuropy_dir = os.path.join(home_dir, 'src', 'neuropy')
sys.path.append(neuropy_dir)
import neuropy as neu


def main():
    """Main function for script."""

    SKIP_EXISTING = True

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
    parser.add_argument('--limit', default=0, type=int)
    args = parser.parse_args()

    # Print arguments
    print('Location: {}'.format(args.location))
    print('Dry run: {}'.format(args.dry_run))
    print('Use parallel processing: {}'.format(args.use_multiproc))
    print('Parallel processing pool size: {}'.format(args.pool_size))

    # Get datasets to process. Convert all with the standard valid criteria.
    EL = neu.el.util.ExperimentLog()
    EL.load()
    EL.get_data_path(args.location)
    criteria = neu.el.util.get_valid_criteria()
    EL.apply_criteria(criteria)

    # Iterate over datasets
    convert_list = []
    for idx, row in EL.log.iterrows():
        # Get path to data
        ds_dir = row['dir_path']
        export_dir = os.path.join(ds_dir, 'exportData')

        # Create pandas directory if needed
        base_dir, _ = os.path.split(ds_dir)
        pandas_dir = os.path.join(base_dir, 'pandasData')
        if not os.path.isdir(pandas_dir):
            print('Creating directory: {}'.format(pandas_dir))
            if not args.dry_run:
                os.makedirs(pandas_dir)

        # There may be multiple files to convert for the dataset. Iterate over
        # these as well
        for ds_name in row['dir_list']:
            # Check for 'SI_translated.mat' file
            file_name = ds_name + '_translated_exportData.mat'
            file_path = os.path.join(export_dir, file_name)
            if not os.path.isfile(file_path):
                # If a '_translated.mat' file does not exist, check for a '.mat'
                # file. This seems to be an issue for some of the earlier SSD
                # datasets, where the files were renamed when copied to the
                # translated directory.
                file_name = ds_name + '_exportData.mat'
                file_path = os.path.join(export_dir, file_name)
                if not os.path.isfile(file_path):
                    # No valid file was found, print a warning message and
                    # proceed to the next iteration.
                    warn_str = (
                        'WARNING: could not find a valid dataset file for '
                        + ds_name
                    )
                    print(warn_str)
                    continue

            # First check to see if the file already exists. If so, skip
            save_name = ds_name + '_pandasData.hdf'
            save_path = os.path.join(pandas_dir, save_name)
            if os.path.isfile(save_path) and SKIP_EXISTING:
                print('File {} exists. Skipping ...'.format(save_name))
                continue

            # Create dict with file info
            file_info = {
                'dry_run': args.dry_run,
                'file_name': file_name,
                'file_path': file_path,
                'save_name': save_name,
                'pandas_dir': pandas_dir
            }
            convert_list.append(file_info)

    # Limit files to convert if desired
    if args.limit > 0:
        n_rows = min(EL.log.shape[0], args.limit)
        convert_list = convert_list[0:n_rows]

    # Now convert files
    if args.use_multiproc:
        with mp.Pool(processes=args.pool_size) as pool:
            pool.map(convert_file, convert_list)
    else:
        for file_info in convert_list:
            convert_file(file_info)

    return None


def convert_file(file_info):
    """Convert MATLAB *.mat file to Pandas HDF5 format."""
    # If the file exists, load and convert
    print('Converting: {}'.format(file_info['file_name']))

    # If running in dry-run mode, do not attempt to convert
    if file_info['dry_run']:
        return

    # Load data, convert to DataFrame, clean
    data = neu.util.convertmat.convert_mat(file_info['file_path'])
    df = pd.DataFrame(data['S'])
    neu.el.proc.clean_trajectory_data(df)

    # Save
    save_path = os.path.join(file_info['pandas_dir'], file_info['save_name'])
    df.to_hdf(save_path, key='df')

    return None


# Call main function
if __name__ == '__main__':
    main()
