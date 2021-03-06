# -*- coding: utf-8 -*-
"""
Split the dataset based on the given ratio.
"""


import numpy as np

def split_data(y, x, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    if ratio<0 or ratio>1:
        raise ValueError("Train ratio must be betwen 0 and 1")
    np.random.seed(seed)
    n = y.shape[0]
    indices = np.random.permutation(n)
    train_lim = int(n*ratio)
    train_idx, test_idx = indices[:train_lim], indices[train_lim:]
    y_train, y_test = y[train_idx], y[test_idx]
    x_train, x_test = x[train_idx,:], x[test_idx,:]
    return (y_train, x_train, y_test, x_test)
