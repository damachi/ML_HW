import numpy as np


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.T.dot(e) / (2 * len(e))
    return mse


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    l = np.log(1 + np.exp(tx@w))
    n = y*(tx@w)
    return (l-n).sum(axis = 0)
