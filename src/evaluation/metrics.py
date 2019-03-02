"""
Metrics used to evaluate results of prediction.
"""
import numpy as np
from scipy import stats


def mean_absolute_error(y, y_hat, axis=None, inds=None):
    if inds is not None:
        return np.mean(np.abs(y[inds] - y_hat[inds]), axis=axis)
    else:
        return np.mean(np.abs(y - y_hat), axis=axis)


def mean_squared_error(y, y_hat, axis=None, inds=None):
    if inds is not None:
        return np.mean(np.power(y[inds] - y_hat[inds], 2), axis=axis)
    else:
        return np.mean(np.power(y - y_hat, 2), axis=axis)


def root_mean_squared_error(y, y_hat, axis=None, inds=None):
    if inds is not None:
        return np.sqrt(np.mean(np.power(y[inds] - y_hat[inds], 2), axis=axis))
    else:
        return np.sqrt(np.mean(np.power(y - y_hat, 2), axis=axis))


def neg_log_likelihood_poisson(y, y_hat, lam=1e-6, axis=None):
    return -np.sum(stats.poisson.logpmf(y, y_hat + lam), axis=axis)
