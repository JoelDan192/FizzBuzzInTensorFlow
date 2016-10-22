# -*- coding: utf-8 -*-
"""cross-validation toolbox"""
import numpy as np

def build_k_indices(num_row, k_fold, seed):
    """build k indices for k-fold."""
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_step(y, x, k_indices, k, lambda_, regression_method, regression_loss):
    """@input: regression_method(y, tx, lambda_)
               regression_loss(y, tx, w)."""
               
    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    test_idx = k_indices[k,:]
    row_idx = list(range(k_indices.shape[0]))
    train_idx = k_indices[row_idx[:k]+row_idx[k+1:],:].flatten()
    y_train, x_train = y[train_idx], x[train_idx, :]
    y_test, x_test = y[test_idx], x[test_idx, :]
    loss_tr, w = regression_method(y_train, x_train, lambda_)
    loss_te = regression_loss(y_test, x_test, w, lambda_)
    return (loss_tr, loss_te)

def cross_validation(y, x, k_fold, lambdas, regression_method, regression_loss):
    """returns one (rmse_tr, rmse_te) e.g. rmse training/test error per lambda, obtained
    via avaraging in a k-fold cross validation"""
    seed = 1
    # split data in k fold
    k_indices = build_k_indices(y.shape[0], k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    for lamb in lambdas:
        e_tr, e_te = zip(*[cross_validation_step(y, x, k_indices, k, lamb, regression_method, regression_loss) for k in range(k_fold)])
        e_tr, e_te = np.array(e_tr).mean(), np.array(e_te).mean()
        rmse_tr.append(e_tr)
        rmse_te.append(e_te)
    return zip(lambdas, rmse_tr, rmse_te)
