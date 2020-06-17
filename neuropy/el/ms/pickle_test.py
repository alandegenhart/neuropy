"""Pickle test script"""

import pickle
import os
import numpy as np

# Generate some fake data
my_dict = {'a': 'A string', 'b': [1, 2, 3, 4], 'c': np.array([1, 2, 3])}

# Set up paths
home_dir = os.path.expanduser('~')
save_dir = os.path.join(home_dir, 'results', 'pickle_test')
os.makedirs(save_dir, exist_ok=True)
save_fname = os.path.join(save_dir, 'my_data.p')

# Save data
file = open(save_fname, 'wb')
pickle.dump(my_dict, file)
file.close()

# Read data
file = open(save_fname, 'rb')
my_dict_in = pickle.load(file)
print(my_dict_in)