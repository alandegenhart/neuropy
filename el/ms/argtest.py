"""Argument test"""

import sys

# Print number of arguments
print('Number of arguments: {}'.format(len(sys.argv)))

# Print each argument
for n, arg in enumerate(sys.argv):
    print('Argument {}: {}'.format(n, arg))