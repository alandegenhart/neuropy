# NeuroPy: a Python toolbox for the analysis of neural data

**Note:** this package is in active development and use and is likely to change dramatically without notice.

This package contains a collection of Python code for the analysis of neural data. It was initially created while I was a postdoc at Carnegie Mellon University to aid in the analysis of intracortical recordings during brain-computer interface (BCI) tasks. After moving to the Allen Institute I continued to use and develop this code base, adding functionality for interacting with [Allen Institute Brain Observatory data](http://observatory.brain-map.org/visualcoding).

## Installation
I recommend installing this package in 'development' mode via pip.  After cloning the repository, navigate to the root directory of the package and run: `pip install -e .`.  I generally prefer using Anaconda for environment management, though this should also work with `venv`.

## Highlights (i.e., things that might be useful to others):
- `analysis/stiefel.py` : code to perform optimization over the stiefel manifold.  Allows the user to create subclasses for new objective functions.
- `dimred.py` : module for dimensionality-reduction analyses.  Contains a Python implementation of Factor Analysis based on Expectation Maximization (EM).
- `pbstools.py` : module for running code on the AIBS cluster (PBS/Moab).

## Brief description of all modules
**Note:** this list is not currently up to date.

- `analysis` : general analysis sub-package
    - `gpfa.py` : module for processing GPFA (Gaussian Process Factor Analysis) data
    - `math.py` : currently a catch-all for math functions without a home.  Contains an implementation of Fisher's LDA. *[deprecated]*
    - `stiefel.py` : code to perform optimization over the stiefel manifold.  Allows the user to create subclasses for new objective functions.
- `array.py` : helper functions for interacting with array data
- `decomposition.py` : outdated code for performing dimensionality-reduction analyses.  Will eventually be merged with `dimred.py`. *[deprecated]*
- `dev.py` : contains functions for interacting with ABIS Brain Observatory data.
- `dimred.py` : module for dimensionality-reduction analyses.  Contains a Python implementation of Factor Analysis based on Expectation Maximization (EM).
- `mathutil.py` : another math catch-all.  Supersedes `analysis/math.py`, which will eventually be removed.
- `movie.py` : code for interacting with AIBS Brain Observatory natural movie data.
- `pbstools.py` : module for running code on the AIBS cluster (PBS/Moab).
- `plot.py` : useful plotting functions.  Some may be analysis-specific currently.
- `preprocessing.py` : preprocessing code for AIBS Brain Observatory data.
- `temp.py` : temporary location for code without a home.  Currently contains a lot of functions specific to a project from my postdoc that will be removed.
- `util` : utilities sub-package
    - `convertmat.py` : module for loading MATLAB HDF5 data into Python using Pandas
- `utilities` : AIBS-related utilities.
