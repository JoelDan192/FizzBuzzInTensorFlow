# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y-np.dot(tx,w)
    n = e.shape[0]
    return (1/(n))*np.dot(e.T, e)
