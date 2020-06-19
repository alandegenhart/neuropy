"""Experiment log test script."""

# Import/setup
import neuropy as neu

# Load log
EL = neu.el.util.ExperimentLog()
EL.load()
EL.get_data_path('ssd')
print(EL.log)

# Specify and apply criteria
criteria = {
    'subject': 'Earl',
    'grid': 'targ_4',
    'int_calib_method': 'center_out_udp',
    'bins': 7,
    'align_bins': 1,
    'align': 'start',
    'gpfa_int': 'grad_train',
    'gpfa_rot': 'cond_4'
}
EL.apply_criteria(criteria)

# Get experiment sets
task_list = ['tt_int', 'tt_rot']
experiment_list = EL.get_experiment_sets(task_list)
print(experiment_list)

# TODO:
# - Iterate over experiments and convert export data to HDF data
# - Will need to check to see why export data isn't being generated locally.
#   This might mean checking for the 'translated' and non-translated flats.
# - Iterate to run flow analysis
# - Move analysis results to NSPG