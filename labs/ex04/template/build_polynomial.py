# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np



def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    transformed_data = []
    for xi in x:
        transformed_data.append(list(map(lambda d: xi**d, range(degree+1))))
    return np.array(transformed_data)
