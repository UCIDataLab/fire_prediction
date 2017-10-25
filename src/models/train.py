"""
Used to run training.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

import poisson_regression as pr
import linear_regression as lr
import evaluation as ev
from helper import date_util as du

def train(X, t_k_arr, leave_one_out=True, years=None):
    # Standardize weather
    for cov in ['temperature', 'humidity', 'wind', 'rain']:
        X[cov] = (X[cov] - np.mean(X[cov])) / np.var(X[cov])

    X = X.assign(year=map(lambda x: x.year, X.date_local))
    
    # Filter years
    if years:
        X = X[X.year.isin(years)]
    
    results_dict = defaultdict(list)
    for t_k in t_k_arr:
        print 'Starting t_k=%d' % t_k

        # Filter out predicting before fire started
        legit_series = pd.Series(index=X.index)
        for clust in X.cluster_id.unique():
            clust_df = X[X.cluster_id==clust]
            legit_day = np.min(clust_df.date_local) + du.INC_ONE_DAY * (t_k+1)
            legit_series[clust_df[clust_df.date_local>=legit_day].index]=1        

        X_legit = X[legit_series==1]

        X_t = pr.PoissonRegressionModel(t_k, []).add_autoregressive_col(X_legit, t_k+1)

        results_dict['baseline'].append((X_t.num_det, X_t.num_det_prev))

        prm = pr.PoissonRegressionModel(t_k=t_k, covariates=[])
        if leave_one_out:
            results, years = ev.cross_validation_years(prm, X_t)
        else:
            results, years = ev.leave_none_out(prm, X_t)
        results_dict['auto'].append(np.concatenate(results, axis=1))

        prm = pr.PoissonRegressionModel(t_k=t_k, covariates=['temperature', 'humidity'])
        if leave_one_out:
            results, years = ev.cross_validation_years(prm, X_t)
        else:
            results, years = ev.leave_none_out(prm, X_t)
        results_dict['temp_humid'].append(np.concatenate(results, axis=1))

        prm = pr.PoissonRegressionModel(t_k=t_k, covariates=['temperature', 'humidity', 'wind', 'rain'])
        if leave_one_out:
            results, years = ev.cross_validation_years(prm, X_t)
        else:
            results, years = ev.leave_none_out(prm, X_t)
        results_dict['all'].append(np.concatenate(results, axis=1)) 
     
        """
        lrm = lr.LinearRegressionModel(t_k=t_k, covariates=[])
        if leave_one_out:
            results, years = ev.cross_validation_years(lrm, X_t)
        else:
            results, years = ev.leave_none_out(prm, X_t)
        results_dict['auto_linear'].append(np.concatenate(results, axis=1))
        
        lrm = lr.LinearRegressionModel(t_k=t_k, covariates=['temperature', 'humidity', 'wind', 'rain'])
        if leave_one_out:
            results, years = ev.cross_validation_years(lrm, X_t)
        else:
            results, years = ev.leave_none_out(prm, X_t)
        results_dict['linear_all'].append(np.concatenate(results, axis=1))
        """
    return results_dict
