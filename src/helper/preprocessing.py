"""
Preprocessing functions for data.
"""
import numpy as np
from helper import date_util as du

def add_autoregressive_col(X, t_k):
    """
    Add an autoregressive column to the data.

    :param t_k: num of days ahead to generate col from (1: tomorrow, ...)
    """
    num_det_target = np.empty(X.shape[0])
    for i, row in enumerate(X.itertuples()):
        date, cluster_id, num_det = row.date_local, row.cluster_id, row.num_det

        cluster_df = X[(X.cluster_id==cluster_id) & (X.date_local==date+du.INC_ONE_DAY*(t_k))]
        val = cluster_df.num_det.iloc[0] if not cluster_df.empty else 0

        num_det_target[i] = val

    return X.assign(num_det_target=num_det_target)

def standardize_covariates(X, covariates):
    for cov in covariates:
        X[cov] = (X[cov] - np.mean(X[cov])) / np.var(X[cov])

    return X
