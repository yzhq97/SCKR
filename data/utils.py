import numpy as np
import os
import re
import math
from bisect import bisect_left

image_extensions = [ '.jpg', '.jpeg', '.png', '.bmp' ]

def parse_data_file_list(data_dir, data_file_list_path, label_start_with_zero=True):
    """
    :param data_dir: directory under which data are placed
    :param data_file_list_path: the path to a data list file
    :return: a list of tuples： (data_file_path, label)
    """
    data_files_with_labels = []
    pattern = re.compile(r"(\S+)_(\d+)_(\d+).(\S+)")
    with open(data_file_list_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            match = pattern.match(line)
            if match:
                start, end = match.regs[3]
                label = int(line[start:end])
                if not label_start_with_zero: label -= 1
                path = os.path.join(data_dir, line)
                data_files_with_labels.append( (path, label) )
            else:
                raise Exception("Invalid file name found in %s" % data_file_list_path)

    return data_files_with_labels

def divide_list(l, n):
    """Divides list l into n successive chunks."""

    length = len(l)
    chunk_size = int(math.ceil(length/n))
    expected_length = n * chunk_size
    chunks = []

    for i in range(0, expected_length, chunk_size):
        chunks.append(l[i:i+chunk_size])

    for i in range(len(chunks), n):
        chunks.append([])

    return chunks

def split_and_pack(data_rows):
    """Reorganize list of rows of data into cols of data, and convert the cols to numpy arrays"""
    n_col = len(data_rows[0])
    data_cols = [[datup[col] for datup in data_rows] for col in range(n_col)]
    data_arrays = [np.array(col) for col in data_cols]
    return data_arrays

def parse_data_file_name(str):
    pattern = re.compile(r"(\S+)_(\d+)_(\d+).(\S+)")
    match = pattern.match(str)
    start, end = match.regs[1]
    series = str[start:end]
    start, end = match.regs[2]
    number = int(str[start:end])
    start, end = match.regs[3]
    label = int(str[start:end])
    return series, number, label

def parse_data_file_name_raw(str):
    pattern = re.compile(r"(\S+)_(\d+)_(\d+).(\S+)")
    match = pattern.match(str)
    start, end = match.regs[1]
    series = str[start:end]
    start, end = match.regs[2]
    number = str[start:end]
    start, end = match.regs[3]
    label = str[start:end]
    return series, number, label

def dedup(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def h5ds_fast_index(ds, indices):
    unique_indices = sorted(set(indices))
    data = ds[unique_indices]
    indices_of_indices = [bisect_left(unique_indices, idx) for idx in indices]
    return data[indices_of_indices]

def humanize_duration(t):
    t_int = int(t)
    hours = t_int / 3600
    minutes = (t_int / 60) % 60
    seconds = t_int % 60
    fractions = int(100 * (t - t_int))
    t_str = "%02d:%02d:%02d.%02d" % (hours, minutes, seconds, fractions)
    return t_str

