# -*- coding: utf-8 -*-
"""A function to compute the cost."""

import numpy as np
def compute_mse(y, tx, beta):
    """compute the loss by mse."""
    e = y - tx.dot(beta)
    mse = np.dot(e.T,e) / (2 * len(e))
    return mse
