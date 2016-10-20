# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
from helpers import batch_iter


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y-np.dot(tx,w)
    n = e.shape[0]
    return (1/(2*n))*np.dot(e.T,e)


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    yn, xn = next(batch_iter(y,tx,1))
    return -(yn-np.dot(xn,w))*xn.T


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        g = None
        for _ in batch_size:
            g+= compute_stoch_gradient(y,tx,w)
        g = 1/(batch_size)*g
        loss = compute_loss(y, tx, w)
        w = w - gamma*g
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws
