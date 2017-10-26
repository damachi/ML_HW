import numpy as np
from hw_helpers import batch_iter
from gradients import *
from costs import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    loss = float('inf')
    for n_iter in range(max_iters):
        gradient = compute_gradient_mse(y,tx,w)
        print(gradient)
        loss = compute_mse(y,tx,w)
        print(loss)
        w = w - gamma*gradient
        ws.append(w)
        print(w)
        losses.append(loss)

    return w, loss


def least_squares_SDG(y, tx, initial_w, max_iters, gamma):
    losses= []
    w = initial_w
    ws = [initial_w]
    losses = []
    loss = float('inf')
    batch_size = 1

    for i in range(max_iters):
        for minbatch_y, minbatch_x in batch_iter(y, tx, batch_size):
            gradient = compute_gradient_mse(minbatch_y, minbatch_x, w)
            #SGD using mse
            loss = compute_mse(minbatch_y, minbatch_x, w)
            w = w-gamma*gradient
            ws.append(w)
            losses.append(loss)
    #returns only the final choice
    return w, loss


def least_squares(y,tx):
    if np.linalg.matrix_rank(tx) == tx.shape[1]:
        w = np.linalg.inv(tx.T@tx)@tx.T@y
        ls = compute_mse(y, tx, w)
        return w, ls
    else:
        w = np.linalg.pinv(tx)@y
        ls = compute_mse(y, tx, w)
        return w, ls



def ridge_regression(y, tx, lambda_):
    w = np.linalg.inv(tx.T@tx + (lambda_*2.0*float(tx.shape[0]))*np.identity(tx.shape[1]))@(tx.T@y)
    #w = np.linalg.inv(tx@tx.T + lambda_*np.identity(tx.shape[1]))@tx@y
    e = compute_mse(y, tx, w)
    return w, e


"""
Logistic regression using gradient descent
"""


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    losses = []
    threshold = 1e-8
    for i in range(max_iters):
        g = compute_gradient_likelihood(y, tx, w)
        w = w-gamma*g
        loss = calculate_loss(y, tx, w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2] < threshold):
            break
    loss = calculate_loss(y, tx, w)
    return w, loss

"""
Regularized logistic regression using gradient descent
"""


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    losses = []
    threshold = 1e-8
    for i in range(max_iters):
        for minbatch_y, minbatch_x in batch_iter(y,tx, 1):
            loss = calculate_loss(minbatch_y, minbatch_x, w) + (lambda_/2)*(w@w.T)
            losses.append(loss)
            g = compute_gradient_likelihood(minbatch_y, minbatch_x, w) + lambda_ * w
            w = w - gamma*g
            if len(losses) > 1 and np.abs(losses[-1]- losses[-2] < threshold):
                break

    return w,loss

"""
Newton's method for logistic regression
"""


def new_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    losses = []
    w = initial_w
    threshold = 1e-8
    for i in range(max_iters):
        loss = calculate_loss(y, tx, w)
        losses.append(loss)
        g = compute_gradient_likelihood(y, tx, w) + lambda_ * w
        hes = calculate_hessian(y, tx, w)
        w = w-gamma*np.linalg.inv(hes)@g
        if len(losses) > 1 and np.abs(losses[-1]- losses[-2] < threshold):
            break
    loss = calculate_loss(y, tx, w)
    return w,loss
