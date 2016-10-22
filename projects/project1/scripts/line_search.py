# -*- coding: utf-8 -*-
"""backtracking line search
Unconstrained Optimization Chapter- Convex Optimization- Boyd and Vandenberghe"""

import numpy as np

def backtracking_line_search(alpha, beta, x, f, gf, deltax):
    if (alpha<0 or alpha>0.5) or (beta<0 or beta>1):
        raise ValueError
    slope = np.dot(gf.T,deltax)
    intercept = f(x)
    t=1.0
    while f(x+t*deltax)>=(intercept+alpha*t*slope):
        t=beta*t
    return t