import argparse
import glob
import os

import numpy as np

from lmpy import Matrix

# Methods (function(pam), name) tuples
METHODS = [
]

# .............................................................................
def load_pams(pam_dir):
    """Loads PAMs from CSV files in the specified directory

    Args:
        pam_dir (str): A directory containing PAM CSV files
    """
    pams = []
    for fn in glob.glob(os.path.join(pam_dir, '*.csv')):
        with open(fn) as in_f:
            pams.append(
                Matrix.load_csv(in_f, num_header_rows=1, num_header_cols=3))
    return pams

# .............................................................................
def speed_test_method(rand_fn, pam, iterations):
    """Measures the mean running time for the provided function and PAM

    Args:
        rand_fn (method): A method that returns a randomized PAM when provided
            an observed PAM.
        pam (Matrix): An observed PAM.
        iterations (int): The number of iterations to perform.

    Returns:
        (float, float): Total running time, average running time.
    """
    a_time = time.time()
    for _ in range(iterations):
        _ = rand_fn(pam)
    b_time = time.time()
    total_time = b_time - a_time
    return (total_time, total_time / iterations)

# .............................................................................
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'pam_dir', type=str, help='A directory containing PAM CSV files')
    parser.add_argument(
        'iterations', type=int,
        help='The number of iterations to perform per test')
    args = parser.parse_args()

    for pam in load_pams(args.pam_dir):
        for rand_fn, func_name in METHODS:
            total_time, avg_time = speed_test_method(
                rand_fn, pam, args.iterations)
            print('{} - {} ({})'.format(func_name, total_time, avg_time))
 
