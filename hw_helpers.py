import numpy as np


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def sigmoid(t):
    return 1/(1 + np.exp(-t))


def standardize(x):
    #print(np.nanmean(x, axis=0))
    centered_data = np.subtract(x, np.nanmean(x, axis=0))
    std_data = centered_data / np.nanstd(centered_data, axis=0)
    return std_data


def standardize_test(x_tr, x_te):
    mean = np.nanmean(x_tr, axis=0)
    centered_data = np.subtract(x_te, mean)
    std = np.nanstd(x_tr, axis=0)
    std_data = centered_data/std
    return std_data


def build_model_data(y, x):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    #poly_matrix = np.zeros((x.shape[0],degree+1))
    #deg = np.arange(degree+1)
    #for i in range(degree+1):
    #    poly_matrix[:,i] = x

    return np.vander(x, degree, increasing=True)
