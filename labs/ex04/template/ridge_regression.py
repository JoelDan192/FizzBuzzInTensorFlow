# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
import costs

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    n = y.shape[0]
    w = np.linalg.solve(np.dot(tx.T,tx)+ lamb*np.eye(tx.shape[1]), np.dot(tx.T,y))
    return (costs.compute_mse(y, tx, w), w)