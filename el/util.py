"""Utilities module for Energy Landscape."""


# Functions to implement here:
# - get paired targets
# - get colors

def data_path(location):
    """Return path to stored data."""
    location = location.lower()
    if location == 'yu':
        path = '/afs/ece.cmu.edu/project/nspg/data/batista/el/Animals'
    elif location == 'batista':
        path = ''  # Not currently set up
    elif location == 'ssd':
        path = '/Volumes/Samsung_T5/Batista/Animals'
    else:
        raise NameError('Invalid data location specified.')

    return path

