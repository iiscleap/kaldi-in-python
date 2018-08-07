import multiprocessing as mp
import numpy as np
import re


def load_array(file_name):
    return np.load(file_name)


def run_parallel(func, args_list, n_workers=10):
    pool = mp.Pool(n_workers)
    out = pool.map(func, args_list)
    pool.close()
    if out is not None:
        return list(out)


def save_array(file_name, obj):
    np.save(file_name, obj)


def scp_to_dict(scp_file):
    scp_dict = dict()
    with open(scp_file, 'r') as f:
        for line in f.readlines():
            tokens = re.split('[\s]+', line.strip())
            scp_dict[tokens[0]] = tokens[1]
    return scp_dict
