"""
Used to evaluate models.
"""

from collections import defaultdict

import numpy as np

from src.evaluation import cross_validation as cv
from src.helper import loaders as ld


def setup_ignition_data(ignition_cube_src, fire_cube_src):
    X_ignition_c = ld.load_pickle(ignition_cube_src)
    Y_detections_c = ld.load_pickle(fire_cube_src)

    # Filter data to correct dates
    fire_season = ((5, 14), (8, 31))
    X_ignition_c = X_ignition_c.filter_dates_per_year(fire_season)
    Y_detections_c = Y_detections_c.filter_dates_per_year(fire_season)

    return X_ignition_c, Y_detections_c


def evaluate_model(model_func, X, y, years, t_k, train=True, predict=True, predict_in_sample=True):
    # Cross validate over years
    if years is not None:
        (results_tr, results_te), models = cv.cv_years(model_func, X, y, years, t_k, train, predict, predict_in_sample)
    else:
        (results_tr, results_te), models = cv.evaluate_all(model_func, X, y, t_k, train, predict)

    results_tr = np.concatenate(results_tr, axis=3)
    results_te = np.concatenate(results_te, axis=3)

    return (results_tr, results_te), models


def evaluate_model_grid(model, X_active_r, X_ignition_c, Y_detections_c, years, t_k):
    # Cross validate over years
    results = cv.cv_years_grid(model, X_active_r, X_ignition_c, Y_detections_c, years, t_k)
    results = np.concatenate(results, axis=3)

    return results


def evaluate_model_params(model_func, param_dict, X, y, years, t_k_arr, train=True, predict=True,
                          predict_in_sample=True):
    results_tr_all = defaultdict(list)
    results_te_all = defaultdict(list)
    models_all = defaultdict(list)

    for t_k in t_k_arr:
        # Test model with different covariates
        print('T_k=%d' % t_k, end='')
        for name, params in param_dict.items():
            (results_tr, results_te), models = evaluate_model(model_func(params), X[t_k], y[t_k], years, t_k,
                                                              train, predict, predict_in_sample)
            results_tr_all[name].append(results_tr)
            results_te_all[name].append(results_te)
            models_all[name].append(models)

    print()
    return (results_tr_all, results_te_all), models_all


def evaluate_model_params_nw(model_func, X, y, years, t_k_arr, train=True, predict=True,
                             predict_in_sample=True):
    results_tr_all = []
    results_te_all = []
    models_all = []

    for t_k in t_k_arr:
        # Test model with different covariates
        print('T_k=%d' % t_k, end='')
        (results_tr, results_te), models = evaluate_model(model_func, X[t_k], y[t_k], years, t_k,
                                                          train, predict, predict_in_sample)
        results_tr_all.append(results_tr)
        results_te_all.append(results_te)
        models_all.append(models)

    print()
    return (results_tr_all, results_te_all), models_all
