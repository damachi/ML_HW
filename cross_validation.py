import numpy as np
from costs import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(method, y, x, k_indices, k, lambda_=0, max_iters=1000, gamma=0.5):
    """return the loss of method."""
    initial_w = np.zeros(x.shape[1])
    x_te = x[k_indices[k]]
    y_te = y[k_indices[k]]
    remain_indices = np.delete(k_indices.copy(), (k), axis = 0)
    remain_indices = remain_indices.ravel()
    x_tr = x[remain_indices]
    y_tr = y[remain_indices]
    weights_tr, e_tr = method(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)
    loss_tr = (2*e_tr)
    loss_te = (2*calculate_loss(y_te, x_te, weights_tr)+ (lambda_/2)*(weights_tr.T@weights_tr))

    return loss_tr, loss_te
