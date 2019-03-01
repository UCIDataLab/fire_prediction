import numpy as np
import luigi
import os
import xarray as xr
import pandas as pd
import datetime as dt
import pickle
import logging
from collections import defaultdict
import uuid

import time

import helper.multidata_wrapper as mdw
from evaluation import metrics
from helper.date_util import filter_fire_season
from evaluation import setup_data_structs as setup_ds, evaluate_model as evm

from models import regression_models, grid_models, forecast_models, zero_inflated_models

from .pipeline_params import GFS_RESOLUTIONS, REGION_BOUNDING_BOXES, WEATHER_FILL_METH
from .dataset_pipeline import GridDatasetGeneration
from .weather_pipeline import WeatherGridGeneration

MODEL_STRUCTURES = ['grid', 'cluster']
MODEL_TYPES = {
        'zero_inflated_p': zero_inflated_models.ZeroInflatedPoissonRegression,
        'hurdle_p': zero_inflated_models.PoissonHurdleRegression, 
        'neg_binomial': regression_models.NegativeBinomialRegression,
        'poisson': regression_models.PoissonRegression, 
        'log_normal': regression_models.LogNormalRegression,
        'linear': regression_models.LinearRegression,
        'persistence': regression_models.PersistenceModel}
SEPARATED_IGNITIONS = ['unified', 'separated', 'active_only']
MEMORY_TYPES = ['none', 'all', 'decay']
DECAY_METHODS = ['fixed', 'learned']
FORECAST_METHODS = ['separate', 'recursive']
LOG_CORRECTION_METHODS = ['add', 'max']

logger = logging.getLogger('pipeline')

def build_single_model(model_type, covariates, log_covariates, params, response_var='num_det_target',
        exclude_params=None):
    if exclude_params is not None:
        covs = list(covariates)
        for exc in exclude_params:
            covs.remove(exc)
            log_covariates.remove(exc)

    model_cls = MODEL_TYPES[model_type]
    model = model_cls(response_var, covs, log_covariates, params['log_correction_type'], 
            params['log_correction_constant'], params['regularization_weight'], params['normalize_params'])

    return model

def add_memory_all(X_ds, dates, mem_cov, memory_length, memory_start=1):
    names = []

    values = np.array(X_ds[mem_cov].values)

    # Add autoregressive memory
    for i in range(memory_start,memory_length+1):
        y_mem = setup_ds.shift_in_time(values, dates, -i, np.zeros)

        name = mem_cov + '_' + str(i)
        names.append(name)

        X_ds.update({name: (('y','x','time'), y_mem)})

    return names

def add_memory_decay(X_ds, mem_cov, memory_length, decay_method, decay_values=None):
    names = []

    if decay_method == 'learned':
        raise NotImplementedError()

    # Compute expon decay
    decay_val = decay_values.get(mem_cov, decay_values['default'])

    vals = np.power(decay_val, range(0, memory_length))
    vals /= np.sum(vals)

    new = np.zeros(X_ds.num_det.shape)
    for i in range(0, memory_length):
        new += X_ds[mem_cov + '_' + str(i+1)] * vals[i]

    name = mem_cov + '_expon'
    X_ds.update({name: (('y','x','time'), new)})

    return [name]

def add_memory(X_grid_dict, mem_cov, is_log_cov, params):
    memory_length = params['memory_length']

    # cov_names will be same for all t_k so we can just use the last set generated
    for X_ds in X_grid_dict.values():
        dates = np.array(list(map(lambda x: pd.Timestamp(x).to_pydatetime().date(), X_ds.time.values)))

        cov_names = add_memory_all(X_ds, dates, mem_cov, memory_length)

        if params['memory_type'] == 'all':
            pass
        elif params['memory_type'] == 'decay':
            decay_method = params['decay_method']
            decay_values = params['decay_values']

            cov_names = add_memory_decay(X_ds, mem_cov, memory_length, decay_method, decay_values)
        else:
            raise NotImplementedError()

    return cov_names

def add_active(X_grid_dict, active_check_days, params):
    for X_ds in X_grid_dict.values():
        # Active check requires memory covariates
        no_memory = params['memory_type'] == 'none'
        memory_length = params['memory_length']
        if no_memory or ((active_check_days-1) > memory_length):
            memory_start = 1 if no_memory else memory_length+1
            dates = np.array(list(map(lambda x: pd.Timestamp(x).to_pydatetime().date(), X_ds.time.values)))
            _ = add_memory_all(X_ds, dates, 'num_det', active_check_days-1, memory_start=memory_start)

        is_active = X_ds.active.values

        for i in range(1,active_check_days):
            vals = X_ds['num_det_' + str(i)].values # Using memory covariates to avoid recomputing
            is_active = np.logical_or(is_active, vals)

        #name = 'active_' + str(act)
        name = 'active'
        X_ds.update({name: (('y','x','time'), is_active)})

def build_covariates(X_grid_dict, params):
    covariates = list(params['covariates'])
    log_covariates = list(params['log_covariates'])

    if params['memory_type'] != 'none':
        for mem_cov in params['memory_covariates']:
            new_covs = add_memory(X_grid_dict, mem_cov, False, params)
            covariates += new_covs
        for mem_cov in params['memory_log_covariates']:
            new_covs = add_memory(X_grid_dict, mem_cov, True, params)
            log_covariates += new_covs

    if params['active_check_days'] > 1:
        add_active(X_grid_dict, params['active_check_days'], params)

    return X_grid_dict, covariates, log_covariates

def setup_data(in_files, start_date, end_date, forecast_horizon, parameters):
    # Load data
    X_grid_dict_nw = {k: xr.open_dataset(target) for (k,target) in in_files.items()}

    # Setup data
    years_train = list(range(start_date.year, end_date.year+1))
    X_grid_dict_nw, covariates, log_covariates = build_covariates(X_grid_dict_nw, parameters)

    logger.debug('Cov.: %s, Log Cov.: %s' % (str(covariates), str(log_covariates)))

    X_grid_dict_nw = {k: filter_fire_season(v, years=years_train) for (k,v) in X_grid_dict_nw.items()}

    # Build y targets
    t_k_arr = list(range(1, forecast_horizon+1))
    y_grid_dict = setup_ds.build_y_nw(X_grid_dict_nw[1]['num_det'].values, X_grid_dict_nw[1].time.values, t_k_arr, 
            years_train)

    # Setup data wrappers for corresponding model structures
    if parameters['forecast_method'] == 'recursive':
        all_ds = [X_grid_dict_nw[k] for k in range(1, forecast_horizon+1)]
        X_grid_dict_nw = {k: mdw.MultidataWrapper(all_ds) for k in X_grid_dict_nw}
    else:
        X_grid_dict_nw = {k: mdw.MultidataWrapper((ds,ds)) for (k,ds) in X_grid_dict_nw.items()}

    return X_grid_dict_nw, y_grid_dict, covariates, log_covariates, years_train

def build_model(covariates, log_covariates, params, t_k):
    """ Select and instantiate model corresponding to params. """

    if params['separated_ignitions'] == 'unified':
        unified_model = build_single_model(params['active_model_type'], covariates, log_covariates, params)
        model = grid_models.UnifiedGrid(unified_model)

    elif params['separated_ignitions'] == 'active_only':
        active_model = build_single_model(params['active_model_type'], covariates, log_covariates, params)
        model = grid_models.ActiveIgnitionGrid(active_model, None)

    elif params['separated_ignitions'] == 'separated':
        active_model = build_single_model(params['active_model_type'], covariates, log_covariates, params)
        ignition_model = build_single_model(params['ignition_model_type'], covariates, log_covariates, params,
                params['ignition_covariates_exclude'])
        model = grid_models.ActiveIgnitionGrid(active_model, ignition_model)

    else:
        raise NotImplementedError()

    if params['forecast_method'] == 'recursive':
        model = forecast_models.RecursiveForecast(model, t_k)

    return model

def build_model_func(covariates, log_covariates, params):
    return lambda t_k: build_model(covariates, log_covariates, params, t_k=t_k)

def create_job_id(train_params_dict):
    return int(uuid.uuid4())

def flat(x):
    return map(lambda x: x.flatten(), x)

def compute_summary_results(results_tr, results_te, X_grid_dict_nw, years, metrics_=[metrics.root_mean_squared_error, 
    metrics.mean_absolute_error]):
    summary_results = defaultdict(dict)

    # Compute overall error metrics
    for i, metric in enumerate(metrics_):    
        x = ['Avg.'] + list(range(1,len(results_tr)+1))
        y = list(map(lambda x: metric(*flat(x)), results_tr))
        y = [np.mean(y)] + y
        summary_results['train'][metric.__name__] = (x,y)

        x = ['Avg.'] + list(range(1,len(results_te)+1))
        y = list(map(lambda x: metric(*flat(x)), results_te))
        y = [np.mean(y)] + y
        summary_results['test'][metric.__name__] = (x,y)

    #ds = filter_fire_season(X_grid_dict_nw[1][0], years=years)
    #active_inds = ds.active.values.flatten()

    # Compute active and igntion error metrics
    if years is not None:
        active_inds = list(map(lambda i: filter_fire_season(X_grid_dict_nw[i][0], years=years).active.values.flatten(), 
            range(1,len(X_grid_dict_nw)+1)))
        ignition_inds = list(map(lambda x: ~x, active_inds))
    else:
        active_inds = list(map(lambda i: X_grid_dict_nw[i][0].active.values.flatten(), 
            range(1,len(X_grid_dict_nw)+1)))
        ignition_inds = list(map(lambda x: ~x, active_inds))

    for inds_name, inds in [('active', active_inds), ('ignition', ignition_inds)]:
        for i, metric in enumerate(metrics_):    
            x = ['Avg.'] + list(range(1,len(results_te)+1))
            y = list(map(lambda x: metric(*flat(x[0]), inds=x[1]), zip(results_te,inds)))
            y = [np.mean(y)] + y
            summary_results['test'][metric.__name__+'_'+inds_name] = (x,y)

    return summary_results

class TrainModel(luigi.Task):
    data_dir = luigi.parameter.Parameter()
    experiment_dir = luigi.parameter.Parameter()

    start_date = luigi.parameter.DateParameter(default=dt.date(2007,1,1))
    end_date = luigi.parameter.DateParameter(default=dt.date(2016,12,31))

    resolution = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS, default='4')
    bounding_box_name = luigi.parameter.ChoiceParameter(choices=REGION_BOUNDING_BOXES.keys(), default='alaska')

    fire_season_start = luigi.parameter.DateParameter(default=dt.date(2007, 5,14))
    fire_season_end = luigi.parameter.DateParameter(default=dt.date(2007, 8, 31))

    model_structure = luigi.parameter.ChoiceParameter(choices=MODEL_STRUCTURES)
    separated_ignitions = luigi.parameter.ChoiceParameter(choices=SEPARATED_IGNITIONS)
    active_model_type = luigi.parameter.ChoiceParameter(choices=list(MODEL_TYPES.keys()))
    ignition_model_type = luigi.parameter.ChoiceParameter(choices=list(MODEL_TYPES.keys()), default=None)
    covariates = luigi.parameter.ListParameter()
    ignition_covariates_exclude = luigi.parameter.ListParameter()
    memory_type = luigi.parameter.ChoiceParameter(choices=MEMORY_TYPES)
    memory_covariates = luigi.parameter.ListParameter()
    memory_log_covariates = luigi.parameter.ListParameter()
    memory_length = luigi.parameter.NumericalParameter(var_type=int, min_value=0, max_value=100)
    decay_method = luigi.parameter.ChoiceParameter(choices=DECAY_METHODS)
    decay_values = luigi.parameter.DictParameter(default=None)
    forecast_method = luigi.parameter.ChoiceParameter(choices=FORECAST_METHODS)
    active_check_days = luigi.parameter.NumericalParameter(var_type=int, min_value=1, max_value=20)
    regularization_weight = luigi.parameter.NumericalParameter(var_type=float, min_value=0, max_value=100, default=None)
    log_correction_type = luigi.parameter.ChoiceParameter(choices=LOG_CORRECTION_METHODS)
    log_correction_constant = luigi.parameter.NumericalParameter(var_type=float, min_value=0, max_value=1)
    log_covariates = luigi.parameter.ListParameter()
    fill_method = luigi.parameter.ChoiceParameter(choices=WEATHER_FILL_METH)
    normalize_params = luigi.parameter.BoolParameter(default=False)

    forecast_horizon = luigi.parameter.NumericalParameter(var_type=int, min_value=1, max_value=10, default=5)
    years_test = luigi.parameter.ListParameter(default=None)

    def requires(self):
        self.t_k_arr = range(1, self.forecast_horizon+1)
        if self.model_structure == 'grid':
            tasks = {k: GridDatasetGeneration(data_dir=self.data_dir, start_date=self.start_date, end_date=self.end_date,
                resolution=self.resolution, bounding_box_name=self.bounding_box_name, fill_method=self.fill_method,
                forecast_horizon=k) for k in self.t_k_arr}
        else:
            raise NotImplementedError('Training cluster models not supported yet')

        return tasks

    def run(self):
        # Setup data
        X_grid_dict_nw, y_grid_dict, covariates, log_covariates, years_train = setup_data(
                {k: v.path for (k,v) in self.input().items()}, self.start_date, self.end_date, self.forecast_horizon, 
                self.train_parameters)
       
        # Train model
        model_func = build_model_func(covariates, log_covariates, self.train_parameters)
        if self.years_test is not None:
            if self.years_test[0] is None:
                years_test = None
            else:
                years_test = self.years_test
        else:
            years_test = years_train

        results, models = evm.evaluate_model_params_nw(model_func, X_grid_dict_nw, y_grid_dict, years_test, 
                self.t_k_arr)

        summary_results = compute_summary_results(results[1], results[1], X_grid_dict_nw, years_test)

        out_dict = {'models': models, 'summary_results': summary_results, 'params': self.train_parameters}

        # Save model and parameters
        with self.output().temporary_path() as temp_output_path:
            with open(temp_output_path, 'wb') as fout:
                pickle.dump(out_dict, fout)

        logger.info('JOB ID: %d -- %s' % (self.job_id, str(self.train_parameters)))
        logger.debug('RESULTS: %s -- %s' % (str(summary_results),str(self.train_parameters)))

    def output(self):
        self.train_parameters = {
                'model_structure': self.model_structure, 
                'separated_ignitions': self.separated_ignitions, 
                'active_model_type': self.active_model_type, 
                'ignition_model_type': self.ignition_model_type, 
                'covariates': self.covariates, 
                'ignition_covariates_exclude': self.ignition_covariates_exclude, 
                'memory_type': self.memory_type, 
                'memory_covariates': self.memory_covariates, 
                'memory_log_covariates': self.memory_log_covariates, 
                'memory_length': self.memory_length,
                'decay_method': self.decay_method,
                'decay_values': self.decay_values,
                'forecast_method': self.forecast_method, 
                'active_check_days': self.active_check_days,
                'regularization_weight': self.regularization_weight,
                'log_correction_type': self.log_correction_type,
                'log_correction_constant': self.log_correction_constant,
                'log_covariates': self.log_covariates,
                'fill_method': self.fill_method,
                'resolution': self.resolution,
                'region': self.bounding_box_name,
                'forecast_horizon': self.forecast_horizon,
                'normalize_params': self.normalize_params}

        #fn = '_'.join(list(map(str, self.train_parameters))) + '.pkl'
        self.job_id = create_job_id(self.train_parameters)
        fn = str(self.job_id) + '.pkl'

        dest_path = os.path.join(self.experiment_dir, fn)

        return luigi.LocalTarget(dest_path)
        
