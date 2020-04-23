# NeuroPy: a Python toolbox for the analysis of intracortical recordings

**Note:** this package is currently undergoing rapid development and is likely to change dramatically.

This package contains a collection of Python code for performing various neural analysis. It is being designed primarily for the analysis of intracortical recordings during brain-computer interface (BCI) tasks. I am developing this package in concert with the migratino of my postdoctoral analysis workflow from MATLAB to Python.

This package is (currently) organized in the following manner:
- `array.py` : module for interacting with array data
- `analysis` : general analysis sub-package
    - `gpfa.py` : module for interacting with GPFA (Gaussian Process Factor Analysis) data
- `util` : utilities sub-package
    - `convertmat.py` : module for loading MATLAB HDF5 data into Python using Pandas
- `el` : collection of analysis code for the EL project
    - `plot.py` : EL-specific plotting functionality
    - `proc.py` : data processing and cleaning
    - `util.py` : assorted EL-specific utilities functions
    - `validation.py` : module used to validate results, mainly relating to the migration from MATLAB to Python