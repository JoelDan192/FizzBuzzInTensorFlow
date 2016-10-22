# -*- coding: utf-8 -*-
"""functions in the logistic regression family"""
import numpy as np
from cross_validation import cross_validation
from line_search import backtracking_line_search
from split_data import split_data

def sigmoid(x):
    "Numerically-stable sigmoid function."
    return np.exp(-np.logaddexp(0, -x))


def calculate_loss(y, tx, w):
    sig = sigmoid(np.dot(tx,w))
    class_one_log_probs = -np.sum(y*np.log(sig))
    class_two_log_probs = -np.sum((1-y)*np.log(1-sig))
    return class_one_log_probs+class_two_log_probs

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return np.dot(tx.T, sigmoid(np.dot(tx,w))-y)

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    Sig_txw = sigmoid(np.dot(tx,w))
    S = (Sig_txw*(1-Sig_txw)).flatten()
    return np.dot((tx.T)*S,tx)

def logistic_regression_tuple(y, tx, w):
    """return the loss, gradient, and hessian."""
    return (calculate_loss(y, tx, w), calculate_gradient(y, tx, w), calculate_hessian(y, tx, w))

def penalized_logistic_regression_tuple(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss(y, tx, w) + lambda_*np.dot(w.T,w)
    g = calculate_gradient(y, tx, w) + 2*lambda_*w
    h = calculate_hessian(y, tx, w) + 2*lambda_*np.eye(w.shape[0])
    return (loss, g, h)

def newton_step(y, tx, w, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss, g, h = logistic_regression_tuple(y, tx, w)
    w = w - gamma*np.dot(np.linalg.inv(h),g)
    return loss, w

def penalized_newton_step(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, g, h = penalized_logistic_regression_tuple(y, tx, w, lambda_)
    delta_w = -np.dot(np.linalg.inv(h),g)
    w0=w
    return (loss, w + gamma*delta_w, w0, 0.5*np.dot(g.T,-delta_w))

def logistic_regression(y, tx, gamma, max_iters):
    # init parameters
    debug_mode = False
    threshold = 1e-8
    loss = None
    loss_prev = None  
    w = np.zeros(tx.shape[1])
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = newton_step(y, tx, w, gamma)
        # log info
        if debug_mode and iter % 500 == 0:
            print("Current iteration={i}, the loss={l}, gn={gn}".format(i=iter, l=loss, gn=np.dot(g.T,g)))
            print("weights={ws}".format(ws=w))
        # converge criteria
        if  loss_prev and np.abs(loss - loss_prev) < threshold:
            break
    return (loss,w)


def reg_logistic_regression(y, tx, lambda_, gamma, max_iters):
    # init parameters
    debug_mode = False
    threshold = 1e-8
    loss_prev = 0
    loss = None
    
    w = np.zeros(tx.shape[1])

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w, w0, gr_norm = penalized_newton_step(y, tx, w, gamma, lambda_)
        # log info
        if debug_mode and iter % 500 == 0:
            print("Current iteration={i}, the loss={l}, gn={gn}".format(i=iter, l=loss, gn=np.dot(g.T,g)))
            print("weights={ws}".format(ws=w))
        # converge criteria
        if gr_norm<threshold:
            w=w0
            break
        if loss<0:
            raise ValueError("Loss functions should be non-negative")
        if np.abs(loss - loss_prev) < threshold:
            w=w0
            break
        loss_prev = loss
    #returns non-reg loss. to compare to non-reg methods.
    return (calculate_loss(y, tx, w), w)

def logistic_auto(y, tx, lambda_, max_iters, initial_w):
    """Logistic regression with Newton's method and backtracking line-search."""
    loss = None
    w = initial_w
    threshold = 1e-8
    loss_prev = 0
    loss = None
    
    def loss_w(w):
        return calculate_loss(y, tx, w) + lambda_*np.dot(w.T,w)   
    for n_iter in range(max_iters):
        g = calculate_gradient(y,tx,w) + 2*lambda_*w
        loss = calculate_loss(y, tx, w) + lambda_*np.dot(w.T,w)
        h = calculate_hessian(y, tx, w) + 2*lambda_*np.eye(w.shape[0])
        delta_w = -np.dot(np.linalg.inv(h),g)
        #check grad(f)~=0 in the quadratic norm of hessian.
        if 0.5*np.dot(g.T,-delta_w) < threshold or (loss_prev and np.abs(loss - loss_prev) < threshold):
            break
        t = backtracking_line_search(0.45, 0.9, w, loss_w, g,delta_w)
        w = w + t*delta_w
        loss_prev = loss
    #we use it for cross-validation so we keep penalized loss
    return (loss, w)

def reg_logistic_regression_auto(y, tx, kfold, max_iters, lambdas):
    
    initial_w = np.zeros(tx.shape[1])
    def curried_logistic(ys, xs, lambda_):
        return logistic_auto(ys, xs, lambda_, max_iters, initial_w)
    def logistic_reg_loss(ys, xs, ws, lambda_):
        return calculate_loss(ys, xs, ws) + lambda_*np.dot(ws.T,ws)
    
    best_lbda = min(cross_validation(y, tx, kfold, lambdas, curried_logistic, logistic_reg_loss),
        key = lambda tup: tup[2])[0]
  
    train_ratio = 1-(1/float(kfold))
    y_train, x_train, y_test, x_test = split_data(y, tx, train_ratio)
    train_loss, w = curried_logistic(y_train, x_train, best_lbda)
    
    #Model selection (kfold) uses cost with lbda, we return un-regularized cost.
    test_loss = calculate_loss(y_test, x_test, w)
    return (test_loss, w)
    
    
    
