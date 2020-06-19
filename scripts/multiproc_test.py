"""Multiprocessing test script"""

# Import
import multiprocessing as mp
import time


def multi_fun(id):
    """Function to test multiprocessing."""

    n_iter = 5
    pause = 1

    for i in range(n_iter):
        print('ID: {}, Iter: {}'.format(id, i))
        time.sleep(pause)


# Define id list
id_list = range(3)

# Run main function -- sequential
t_start = time.time()
for id in id_list:
    multi_fun(id)

t_end = time.time()
print('Elapsed time: {:0.3f} seconds'.format(t_end - t_start))

# Run main function -- using multiprocessing
t_start = time.time()
with mp.Pool(processes=3) as pool:
    pool.map(multi_fun, id_list)

t_end = time.time()
print('Elapsed time: {:0.3f} seconds'.format(t_end - t_start))