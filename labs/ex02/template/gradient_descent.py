# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
import numpy as np
import costs


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y-np.dot(tx,w)
    n = e.shape[0]
    return (1/(2*n))*np.dot(e.T,e)


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - np.dot(tx,w)
    n = e.shape[0]
    return -(1/n)*np.dot(tx.T,e)

def gradient_descent(y, tx, initial_w, max_iters, gamma): 
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        g = compute_gradient(y,tx,w)
        loss = compute_loss(y, tx, w)
        w = w - gamma*g
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
