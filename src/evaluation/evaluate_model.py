"""
Used to evaluate models.
"""

import click
import numpy as np
import logging
from collections import defaultdict

import metrics 
import cross_validation as cv

from helper import loaders as ld
from helper import preprocessing as pp

from models.poisson_regression import PoissonRegressionModel
from models.grid_predictor import GridPredictorModel
from models.bias_grid import BiasGridModel
from models.active_ignition_grid import ActiveIgnitionGridModel

def setup_active_fire_data(integrated_cluster_df_src, covariates=['temperature', 'humidity', 'wind', 'rain']):
    X_active_df = ld.load_pickle(integrated_cluster_df_src)

    # Preprocess data
    X_active_df = pp.standardize_covariates(X_active_df, covariates)
    X_active_df = X_active_df.assign(year=map(lambda x: x.year, X_active_df.date_local))

    return X_active_df 


def setup_ignition_data(ignition_cube_src, fire_cube_src):
    X_ignition_c = ld.load_pickle(ignition_cube_src)
    Y_detections_c = ld.load_pickle(fire_cube_src)

    # Filter data to correct dates
    fire_season=((5,14), (8,31))
    X_ignition_c = X_ignition_c.filter_dates_per_year(fire_season)
    Y_detections_c = Y_detections_c.filter_dates_per_year(fire_season)

    return X_ignition_c, Y_detections_c


def evaluate_model(model, X, y, years, t_k):
    # Cross validate over years
    (results_tr,results_te), models = cv.cv_years(model, X, y, years, t_k)

    results_tr = np.concatenate(results_tr, axis=3)
    results_te = np.concatenate(results_te, axis=3)

    return (results_tr,results_te), models

def evaluate_model_grid(model, X_active_r, X_ignition_c, Y_detections_c, years, t_k):
    # Cross validate over years
    results = cv.cv_years_grid(model, X_active_r, X_ignition_c, Y_detections_c, years, t_k)
    results = np.concatenate(results, axis=3)

    return results

def evaluate_model_params(model_func, param_dict, X, y, years, t_k_arr):
    results_tr_all = defaultdict(list)
    results_te_all = defaultdict(list)
    models_all = defaultdict(list)

    for t_k in t_k_arr:
        results_k = {}
        
        # Test model with different covariates
        print 'T_k=%d' % t_k,
        for name,params in param_dict.iteritems():
            (results_tr,results_te), models = evaluate_model(model_func(params), X[t_k], y[t_k], years, t_k)
            results_tr_all[name].append(results_tr)
            results_te_all[name].append(results_te)
            models_all[name].append(models)
    
    print
    return (results_tr_all,results_te_all), models_all


@click.command()
@click.argument('integrated_cluster_df_src', type=click.Path(exists=True))
@click.argument('ignition_cube_src', type=click.Path(exists=True))
@click.argument('fire_cube_src', type=click.Path(exists=True))
@click.option('--log', default='INFO')
def main(integrated_cluster_df_src, ignition_cube_src, fire_cube_src, log):
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging, log.upper()), format=log_fmt)

    # Setup data
    X_active_df = setup_active_fire_data(integrated_cluster_df_src)
    X_ignition_c, Y_detections_c = setup_ignition_data(ignition_cube_src, fire_cube_src)

    # Build test model
    logging.info('Building test model')
    afm = GridPredictorModel(PoissonRegressionModel(['temperature', 'humidity', 'wind', 'rain']))
    igm = BiasGridModel()
    model = ActiveIgnitionGridModel(afm, igm)

    # Start training
    logging.info('Starting evaluation')
    evaluate_model(model, X_active_df, X_ignition_c, Y_detection_c)

if __name__=='__main__':
    main()
