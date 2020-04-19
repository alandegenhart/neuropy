"""MATLAB *.mat to Python file conversion.

MATLAB *.mat file structure information:

Structure arrays are organized into groups. The entire structure array object
is a single group. Each field of the structures that make up the structure
array are keys in that group.

my_file = h5py.File(path_to_file, 'r')
my_file.keys()

If the data is saved as a structure array, there will be a single high-level
group with the variable name of the structure array in MATLAB. There is also a
'#refs#' key, which is a group of references to each unique dataset. These
reference names don't make much sense, so we can ignore them for now.

Accessing the associated values for the keys will return datasets, which are of
the shape of the structure array itself and contain the data for a specific
field in the structure array.

ds = my_file['struct_array_name']['field_name']  # Returns a dataset
ds.shape  # Returns the size of the structure array

The size of the datasets appear to be (at least) 2D even if the original data
is only a 1D structure array.

Accessing an individual element in this dataset will return an object
reference:

ref = my_file['struct_array_name']['field_name'][0, 1]

Note that this does still not contain data -- it is a reference to where the
actual data exists. To access the data, we can simply use the reference as a
key for the HDF5 object:

ds = my_hdf5_file[ref]

This returns another dataset, but one that contains the actual data (and not a
reference). To access individual elements in the data, you can index through
them like any array:

element = my_hdf5_file[ref][i, j]

Note that the above might not be the easiest way to access the data, especially
if the data is being converted into a more usable format (e.g., a dataframe).
To get the data only, the 'values' attribute of the dataset object can be used:

data = my_hdf5_file[ref][i, j].value

For most data types (one exception being structures), this will return a numpy
array. This is the case even for strings, which exist as an array of type
uint16. In order to convert these values to characters, the chr() function can
be used:

my_string = ''.join([chr(el) for el in data])

One issue that can arise here is how best to determine if the values for a
particular dataset are actual MATLAB arrays/matrices, or whether they are
strings and need to be converted as described above. To determine this, the
attributes for a dataset can be used. MATLAB records the data type stored in
the dataset in the 'MATLAB_class' attribute:

matlab_class = my_hdf5_file[ref][i, j].attrs['MATLAB_class']

Common data types are:

b'double'
b'char'
b'struct'

Using these data types, it should be fairly straightforward to convert the
data into the appropriate Python data types.

If the data contained in the MATLAB structure field, the dataset will contain
object references in exactly the same manner as the top-level structure. This
means that the same procedure can be run recursively until all of the data has
been converted/accessed.

"""


# Import modules
import h5py
import os
import numpy as np


def convert_mat(path):
    """Convert MATLAB *.mat file to Python dict/list format.

    Inputs:
    path -- path to the file to convert

    This function converts the file specified by PATH to a dict/list format
    that is more easily used.

    Data contained in the HDF5 format can be converted to a dict of lists,
    which can easily be converted into a Pandas dataframe. The final dict
    should have the following structure:

    data = {
        'field_1': [item_1, item_2, ..., item_n],
        'field_2': [item_1, item_2, ..., item_n]
    }

    This can easily be converted into a dataframe in the following way:

    import pandas as pd
    df = pd.DataFrame(data)

    """
    
    # Open file and convert the top-level fields. The 'convert_hdf5_group'
    # function can be used here b/c the top-level structure is identical to
    # that of a group.
    with h5py.File(path, 'r') as f:
        data = convert_hdf5_group(f, f)

    return data


def convert_hdf5_group(file, group):
    """Convert HDF5 group to dict format.

    Inputs:
    file -- HDF5 file object
    group -- HDF5 group to convert

    """
    return {key:convert_hdf5_value(file, group[key])
        for key in group.keys() if key != '#refs#'}


def convert_hdf5_dataset(file, dataset):
    """Convert HDF5 dataset to list.

    Inputs:
    file -- HDF5 file object
    dataset -- HDF5 dataset to convert
    
    All MATLAB data is at least two-dimensional. This makes conversion a bit
    problematic for the desired conversion scheme, which assumes that the
    data contains structure *arrays* which are one-dimensional. If this is
    not the case (e.g., the data contains a 2D array of structure), this
    method of conversion will flatten the n-d array to a 1d array before
    converting.

    """

    # Check datatype. If the datatype is 'h5py.ref_dtype', then the dataset
    # contains references to data. If not, it is likely of a 'normal' datatype
    # and can be read as a list.
    if dataset.dtype is h5py.ref_dtype:
        # Iterate over elements in the dataset. In most cases this means that
        # data in a structure array is being accessed.
        data_list = [convert_hdf5_value(file, file[ref]) 
            for ref in dataset.value.flatten()
            ]
    else:
        data_list = dataset.value

        # Handle special case where the list represents a string
        if dataset.attrs['MATLAB_class'].decode() == 'char':
            data_list = ''.join([chr(x) for x in data_list])

    return data_list


def convert_hdf5_value(file, value):
    """Convert value from HDF5 dataset."""

    # Check input value type and take the appropriate action
    if isinstance(value, h5py.Group):  # Value is a group, convert to dict
        data_value = convert_hdf5_group(file, value)
    elif isinstance(value, h5py.Dataset):  # Value is a dataset, convert to list
        data_value = convert_hdf5_dataset(file, value)
    else:
        ValueError

    return data_value


