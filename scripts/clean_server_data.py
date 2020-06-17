"""Organize Energy Landscape data files."""

import os
import sys
import shutil
import re
import argparse

# Custom modules
import neuropy as neu
home_dir = os.path.expanduser('~')
src_dir = os.path.join(home_dir, 'src')


def main():
    """Main analysis function"""

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
    args = parser.parse_args()

    print('Dry-run: {}'.format(args.dry_run))

    # Define flags
    flags = {
        'DRYRUN': args.dry_run,
        'REMOVE_DIRS': ['trialData', 'trajectoryData', 'exportData'],
        'REMOVE_ORIGDIR': True
    }

    # Define data base directory
    dir_base = neu.el.util.data_path(args.location)
    valid_subjects = ['Earl', 'Dwight']

    # Iterate over subjects
    for s in valid_subjects:
        # Get year directories
        subject_dir_path = os.path.join(dir_base, s)
        _, year_dirs, _ = next(os.walk(subject_dir_path))

        # Iterate over year directories
        for yd in year_dirs:
            # Get month directories
            year_dir_path = os.path.join(subject_dir_path, yd)
            _, month_dirs, _ = next(os.walk(year_dir_path))

            # Iterate over month directories
            for md in month_dirs:
                # Get dataset directories
                month_dir_path = os.path.join(year_dir_path, md)
                _, dataset_dirs, _ = next(os.walk(month_dir_path))

                # Get valid directories (YYYYMMDD)
                dataset_dirs = [
                    dd for dd in dataset_dirs if dd[0:6] == yd + md
                ]
                for dd in dataset_dirs:
                    dataset_dir_path = os.path.join(month_dir_path, dd)
                    clean_dataset_directory(dataset_dir_path, flags)

    return None


def clean_dataset_directory(dir, flags):
    """Clean dataset directory."""
    # Display status message
    print('\nCleaning directory: {}'.format(dir))

    # Define path to translated directory and create if it does not exist
    translated_dir = os.path.join(dir, 'translated')
    os.makedirs(translated_dir, exist_ok=True)

    # First remove directories
    for rm_dir in flags['REMOVE_DIRS']:
        rm_dir_path = os.path.join(translated_dir, rm_dir)
        remove_directory(rm_dir_path, flags['DRYRUN'])

    # Get list of sub-directories
    _, sub_dir_list, _ = next(os.walk(dir))
    exp = re.compile('\d\d_')
    sub_dir_list = [sd for sd in sub_dir_list if exp.match(sd)]

    # Copy data to 'translated' directory
    for sd in sub_dir_list:
        # Get files in the sub directory that end in '_SI.mat'
        sub_dir_path = os.path.join(dir, sd)
        _, _, sd_files = next(os.walk(sub_dir_path))
        exp = re.compile('_SI_translated.mat')
        sd_files = [sdf for sdf in sd_files if exp.findall(sdf)]

        # Move translated files
        for sdf in sd_files:
            # Define paths and copy file if not in 'dry run' mode
            sd_file_path = os.path.join(sub_dir_path, sdf)
            dest_path = os.path.join(translated_dir, sdf)
            print('Copying: {}'.format(sd_file_path))
            if not flags['DRYRUN']:
                shutil.copy2(sd_file_path, dest_path)

        # Remove original directory (if desired)
        if flags['REMOVE_ORIGDIR']:
            remove_directory(sub_dir_path, flags['DRYRUN'])

    return None


def remove_directory(dir, dry_run=False):
    """Remove directory and print message."""
    print('Removing: {}'.format(dir))
    if not dry_run:
        shutil.rmtree(dir, ignore_errors=True)

    return None


if __name__ == '__main__':
    main()
