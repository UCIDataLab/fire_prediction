"""
Metrics used to evaluate results of prediction.
"""
import numpy as np

def mean_absolute_error(y, y_hat):
    return np.mean(np.abs(y - y_hat))
