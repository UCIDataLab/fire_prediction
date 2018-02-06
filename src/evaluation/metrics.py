"""
Metrics used to evaluate results of prediction.
"""
import numpy as np

def mean_absolute_error(y, y_hat, axis=None):
    return np.mean(np.abs(y - y_hat), axis=axis)

def mean_squared_error(y, y_hat, axis=None):
    return np.mean(np.power(y-y_hat,2), axis=axis)

def root_mean_squared_error(y, y_hat, axis=None):
    return np.sqrt(np.mean(np.power(y-y_hat,2), axis=axis))
