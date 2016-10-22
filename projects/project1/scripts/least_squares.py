# -*- coding: utf-8 -*-
"""functions in the least_squares regression family"""

import numpy as np
import costs
from line_search import backtracking_line_search
from helpers import batch_iter

    
def least_squares(y, tx):
    """calculate the least squares solution using normal equations."""
    w = np.linalg.solve(np.dot(tx.T,tx), np.dot(tx.T,y))
    return (costs.compute_loss(y, tx, w), w)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - np.dot(tx,w)
    n = e.shape[0]
    return -(1/n)*np.dot(tx.T,e)
        
def gradient_descent(y, tx, initial_w, max_iters, gamma): 
    """Gradient descent algorithm with interface as asked in
    the project description"""
    loss = None
    w = initial_w
    threshold = 1e-8
    losses = []
    
    for n_iter in range(max_iters):
        g = compute_gradient(y,tx,w)
        if np.sum(np.isnan(g))>0:
            print(w)
            print(g)
            return
        loss = costs.compute_loss(y, tx, w)
        w = w - gamma*g
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return (losses[-1], w)



def least_squares_GD_auto(y, tx, max_iters, initial_w): 
    """Gradient descent algorithm with backtracking line-search"""
    loss = None
    w = initial_w
    threshold = 1e-8
    
    def loss_w(w):
        return costs.compute_loss(y, tx, w)   
    for n_iter in range(max_iters):
        g = compute_gradient(y,tx,w)
        loss = costs.compute_loss(y, tx, w)
        if np.dot(g.T,g)<threshold:
            break      
        t = backtracking_line_search(0.4, 0.8, w, loss_w, g,-g)
        w = w - t*g       
    return (loss, w)

def least_squares_SGD_simple(y, tx, max_iters, batch_size, initial_w, gamma): 
    """Stochastic gradient descent"""
    loss = None
    loss_prev = None
    w = initial_w
    threshold = 1e-8
    
    def loss_w(w):
        return costs.compute_loss(y, tx, w)   
    for n_iter in range(max_iters):
        g = compute_stoch_gradient(y,tx,w,batch_size)
        loss_prev = loss
        loss = costs.compute_loss(y, tx, w)
        delta_w = -g
        if loss_prev and np.abs(loss-loss_prev) < threshold:
            return (loss, w)
        w = w + gamma*delta_w       
    return (loss, w)


def compute_stoch_gradient(y, tx, w, batch_size):
    """Compute a stochastic gradient for batch data."""
    yn, xn = next(batch_iter(y,tx,batch_size))
    residuals_batch = np.array(list(map(lambda tup: -(tup[0]-np.dot(tup[1],w.T)), zip(yn,xn))))
    return (1/batch_size)*np.dot(residuals_batch, xn)


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm with interface as asked in
    the project description"""
    # Define parameters to store w and loss
    loss = None
    w = initial_w
    threshold = 1e-8
    losses = []
    for n_iter in range(max_epochs):
        g = compute_stoch_gradient(y,tx,w,batch_size)
        loss = costs.compute_loss(y, tx, w)
        w = w - gamma*g
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return (losses[-1], w)


def least_squares_GD(y, tx, gamma, max_iters):
    
    w0 = np.zeros(tx.shape[1])
    return gradient_descent(y, tx, w0, max_iters, gamma)
    
def least_squares_SGD(y, tx, gamma, max_iters):
    
    w0 = np.zeros(tx.shape[1])
    batch_size = 1
    return stochastic_gradient_descent(
        y, tx, w0, batch_size, max_iters, gamma)
    

