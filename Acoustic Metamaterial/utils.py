"""
Author(s): Wei Chen (wchen459@gmail.com)
"""

import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})


def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        
    
def safe_remove(pathname):
    if os.path.isdir(pathname):
        shutil.rmtree(pathname)
    elif os.path.isfile(pathname):
        os.remove(pathname)


def show_wall_time(func):
    def wrapper(*args, **kwargs):
        begin = time.time()
        func(*args, **kwargs)
        end = time.time()
        print("Wall-clock time for {}: {:.1f}s".format(func.__name__, end-begin))
    return wrapper


def get_overlap_freq(fre_ranges, frequencies):
    n_designs, n_fre_ranges = fre_ranges.shape[:2]
    n_fre = frequencies.shape[0]
    overlap = np.zeros((n_designs, n_fre, n_fre_ranges))
    for i in range(n_fre_ranges):
        if frequencies.ndim == 1:
        
        # Code 1: 
        # bandgap will be underestimated because overlap = 1 only when bandgaps fully contains the frequency interval
        
            overlap[:, :, i] = np.logical_and(fre_ranges[:, i, :1] <= frequencies[np.newaxis, :], 
                                              fre_ranges[:, i, 1:] >= frequencies[np.newaxis, :])
        else:
            overlap[:, :, i] = np.logical_and(fre_ranges[:, i, :1] <= frequencies[:, :1].T, 
                                              fre_ranges[:, i, 1:] >= frequencies[:, 1:].T)
        
        # Code 2: 
        # bandgap will be overestimated because overlap = 1 as long as bandgaps intersects with the frequency interval
        
        #     overlap[:, :, i] = np.logical_and(fre_ranges[:, i, :1] <= frequencies[np.newaxis, :], 
        #                                       fre_ranges[:, i, 1:] >= frequencies[np.newaxis, :])
        # else:
        #     overlap[:, :, i] = np.logical_and(fre_ranges[:, i, :1] <= frequencies[:, 1:].T, 
        #                                       fre_ranges[:, i, 1:] >= frequencies[:, :1].T)
        
        # frequencies[:, :1], fre_ranges[:, i, :1] is the lower bound of the frequency intervals and bandgaps, respectively
        # frequencies[:, 1:], fre_ranges[:, i, 1:] is the upper bound of the frequency intervals and bandgaps, respectively
    overlap = np.any(overlap, axis=-1)
    return overlap

def get_overlap_k(k_ranges, wavevectors):
    n_designs, n_fre_ranges = k_ranges.shape[:2]
    n_fre = wavevectors.shape[0]
    overlap = np.zeros((n_designs, n_fre, n_fre_ranges))
    for i in range(n_fre_ranges):
        if wavevectors.ndim == 1:
        
        # Code 1: 
        # bandgap will be underestimated because overlap = 1 only when bandgaps fully contains the frequency interval
        
            overlap[:, :, i] = np.logical_and(k_ranges[:, i, :1] <= wavevectors[np.newaxis, :], 
                                              k_ranges[:, i, 1:] >= wavevectors[np.newaxis, :])
        else:
            overlap[:, :, i] = np.logical_and(k_ranges[:, i, :1] <= wavevectors[:, :1].T, 
                                              k_ranges[:, i, 1:] >= wavevectors[:, 1:].T)
        
        # Code 2: 
        # bandgap will be overestimated because overlap = 1 as long as bandgaps intersects with the frequency interval
        
        #     overlap[:, :, i] = np.logical_and(fre_ranges[:, i, :1] <= frequencies[np.newaxis, :], 
        #                                       fre_ranges[:, i, 1:] >= frequencies[np.newaxis, :])
        # else:
        #     overlap[:, :, i] = np.logical_and(fre_ranges[:, i, :1] <= frequencies[:, 1:].T, 
        #                                       fre_ranges[:, i, 1:] >= frequencies[:, :1].T)
        
        # frequencies[:, :1], fre_ranges[:, i, :1] is the lower bound of the frequency intervals and bandgaps, respectively
        # frequencies[:, 1:], fre_ranges[:, i, 1:] is the upper bound of the frequency intervals and bandgaps, respectively
    overlap = np.any(overlap, axis=-1)
    return overlap

def sample_grouped_train_data(train_input, train_output, train_fraction, group_size, random_state=42):

    n_total_groups = len(train_input) // group_size
    n_sample_groups = int(train_fraction * n_total_groups)

    np.random.seed(random_state)
    selected_group_indices = np.random.choice(n_total_groups, size=n_sample_groups, replace=False)

    selected_row_indices = []
    for group_idx in selected_group_indices:
        start_idx = group_idx * group_size
        selected_row_indices.extend(range(start_idx, start_idx + group_size))

    train_input_sampled = train_input[selected_row_indices]
    train_output_sampled = train_output[selected_row_indices]

    return train_input_sampled, train_output_sampled

def grouped_train_test_split(
    input_data,
    output_data, 
    train_fraction,
    group_size,
    random_state=42
):

    n_rows = len(input_data)
    n_full_groups = n_rows // group_size

    group_indices = np.arange(n_full_groups)

    rng = np.random.RandomState(random_state)
    rng.shuffle(group_indices)

    n_train_groups = int(round(train_fraction * n_full_groups))
    train_groups = np.sort(group_indices[:n_train_groups])
    test_groups  = np.sort(group_indices[n_train_groups:])

    def groups_to_rows(groups):
        rows = []
        for g in groups:
            start = g * group_size
            rows.extend(range(start, start + group_size))
        return rows

    train_rows = groups_to_rows(train_groups)
    test_rows  = groups_to_rows(test_groups)

    X_train = input_data[train_rows]
    y_train = output_data[train_rows]
    X_test  = input_data[test_rows]
    y_test  = output_data[test_rows]

    return X_train, y_train, X_test, y_test, train_rows, test_rows