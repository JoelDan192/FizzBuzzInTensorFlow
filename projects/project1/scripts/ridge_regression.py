# -*- coding: utf-8 -*-
"""ridge regression using normal equations"""

import costs
import numpy as np
from cross_validation import cross_validation
from split_data import split_data

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    n = y.shape[0]
    w = np.linalg.solve(np.dot(tx.T,tx)+ lambda_*np.eye(tx.shape[1]), np.dot(tx.T,y))
    return (costs.compute_loss(y, tx, w), w)

def ridge_regression_auto(y, tx, kfold, lambdas):
    """ridge regression with kfold cross-validation"""
    
    #this loss is used with data split in cross_validation. keep parameters generic
    def ridge_loss(ys, xs, ws, lambda_):
        return costs.compute_loss(ys, xs, ws) + lambda_*np.dot(ws.T,ws)
        
    #choose lambda according to avg test error across folds
    best_lbda = max(cross_validation(y, tx, kfold, lambdas, ridge_regression, ridge_loss)
                    ,key=lambda tup: tup[2])[0]
    train_ratio = 1-(1/float(kfold))
    y_train, x_train, y_test, x_test = split_data(y, tx, train_ratio)
    train_mse, w = ridge_regression(y_train, x_train, best_lbda)
    #Model selection (kfold) uses cost with lbda, we return un-regularized cost.
    test_mse = costs.compute_loss(y_test, x_test, w)
    return (test_mse, w)