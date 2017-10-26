import numpy as np
from hw_helpers import sigmoid


def compute_gradient_mse(y, tx, w):
    N = y.shape[0]
    e = y-(tx@w)
    return -1/N*(tx.T@(e))


def compute_gradient_likelihood(y, tx, w):
    return tx.T@(sigmoid(tx@w)-y)


def calculate_hessian(y, tx, w):

    S = np.diag((sigmoid(tx@w)*(1-sigmoid(tx@w))).T[0])

    return tx.T@S@tx
