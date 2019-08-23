"""
"""
import sys
sys.path.append('C:\\Users\\cj\\Documents\\GitHub\\lmpy')
from copy import deepcopy
import numpy as np
from random import sample

import lmpy
from lmpy.randomize.grady import *
from lmpy.randomize.curveball_pub import curve_ball, find_presences
from lmpy.randomize.swap import swap_randomize, trial_swap

METHODS = [
    (lambda x: grady_randomize(x, approximation_heuristic=all_zeros_heuristic), 'grady_all_zeros'),
    (lambda x: grady_randomize(x, approximation_heuristic=all_ones_heuristic), 'grady_all_ones'),
    (lambda x: grady_randomize(x, approximation_heuristic=fill_shuffle_reshape_heuristic), 'grady_fill_shuffle'),
    (lambda x: grady_randomize(x, approximation_heuristic=min_col_or_row_heuristic), 'grady_min_col_row'),
    (lambda x: grady_randomize(x, approximation_heuristic=max_col_or_row_heuristic), 'grady_max_col_row'),
    (lambda x: grady_randomize(x, approximation_heuristic=total_fill_percentage_heuristic), 'grady_total_fill'),
    (lambda x: trial_swap(x, num_trials=int(2 * x.size / (x.sum() / x.size))), 'trial_swap'),
    (lambda x: swap_randomize(x, num_swaps=int(x.sum() / 4)), 'swap'),
    (lambda x: curve_ball(x, find_presences(x)), 'curveball')
]


# ............................................................................
def generate_random_matrix(side_length, coef_variation):
    """Generates a random matrix

    Args:
        side_length (int): The length of one side of the matrix.
        coef_variation (int or float): The coeficient of variation in row fill
    """
    mtx = generate_structured_matrix(side_length, coef_variation)
    for i in range(mtx.size):
        col_1, col_2 = sample(range(side_length), 2)
        mtx[[col_1, col_2], :] = mtx[[col_2, col_1], :]
    for i in range(side_length):
        tmp = mtx[i]
        np.random.shuffle(tmp)
        mtx[i] = tmp
    return mtx

# ............................................................................
def generate_structured_matrix(side_length, coef_variation):
    """Generates a structured matrix

    Args:
        side_length (int): The length of one side of the matrix.
        coef_variation (int or float): The coeficient of variation in row fill
    """
    half_side = int(side_length / 2)
    if coef_variation == 0:
        scaled_vals = []
        for i in range(half_side):
            scaled_vals.append(0)
        for i in range(half_side):
            scaled_vals.append(half_side)
    else:
        target_mean = int(side_length / 4)
        target_std = target_mean * coef_variation / 100
        o_vals = sorted(np.random.normal(target_mean, target_std, side_length))
        min_val = o_vals[0]
        o_range = o_vals[-1] - o_vals[0]
        t_range = side_length / 2
        scaled_vals = [int(t_range * (v - min_val) / o_range) for v in o_vals]
    mtx = lmpy.Matrix(np.ones((side_length, side_length), dtype=np.int8))
    for i in range(len(scaled_vals)):
        mtx[half_side - scaled_vals[i] : side_length - scaled_vals[i], i] = 0
    return mtx

# ............................................................................
def print_mtx(mtx):
    """Print the matrix

    Args:
        mtx : A matrix object to be printed
    """
    for row in mtx:
        print(''.join([str(c) for c in row]))

def calculate_c_score(mtx):
    rs = mtx.sum(axis=0)
    c = 0.0
    p = mtx.shape[1] * (mtx.shape[1] - 1) / 2.0
    for j in range(mtx.shape[1]):
        for i in range(0, j):
            sij = len(np.where(mtx[:, i] + mtx[:, j] == 2)[0])
            cij = (rs[i] - sij) * (rs[j] - sij)
            c += cij
    return c / p

# ............................................................................
def assess_type_i_error(num_iterations, num_tests, method_tuples,
                        side_length, cv):
    """
    """
    counts = {}
    for _, method_name in method_tuples:
        counts[method_name] = 0

    for ii in range(num_iterations):
        o_mtx = generate_random_matrix(side_length, cv)
        obs_c = calculate_c_score(o_mtx)
        for method_fn, method_name in method_tuples:
            c_plus = 0
            for jj in range(num_tests):
                try:
                    r_mtx = method_fn(deepcopy(o_mtx))
                    r_c_score = calculate_c_score(r_mtx)
                    if r_c_score >= obs_c:
                        c_plus += 1
                except Exception as e:
                    #print(e)
                    #print('iteration: {}'.format(ii))
                    #print('test: {}'.format(jj))
                    #raise e
                    pass
            c_per = c_plus / num_tests
            if c_per <= 0.05 or c_per >= 0.95:
                counts[method_name] += 1
    for _, method_name in method_tuples:
        print('{} : {}'.format(method_name, counts[method_name] / num_iterations))

# ............................................................................
if __name__ == '__main__':
    side_lengths = [10, 100, 1000]
    num_iterations = 100
    num_tests = 100
    cv = [0, 1, 10]
    for sl in side_lengths:
        for cv in [0, 1, 10]:
            print('Side length: {}, cv: {}'.format(sl, cv))
            assess_type_i_error(num_iterations, num_tests, METHODS, sl, cv)
            print('')
