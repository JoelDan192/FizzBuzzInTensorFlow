# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
import costs

def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    w = np.linalg.solve(np.dot(tx.T,tx), np.dot(tx.T,y))
    return costs.compute_mse(y, tx, w), w