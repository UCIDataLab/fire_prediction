import datetime as dt
import logging
import os
import pickle
import uuid
from collections import defaultdict

import helper.multidata_wrapper as mdw
import luigi
import numpy as np
import pandas as pd
import scipy
import xarray as xr
from evaluation import metrics
from evaluation import setup_data_structs as setup_ds, evaluate_model as evm
from helper.date_util import filter_fire_season
from helper.geometry import get_default_bounding_box
from models import regression_models, grid_models, forecast_models, zero_inflated_models, mlp

from .dataset_pipeline import GridDatasetGeneration
from .pipeline_params import GFS_RESOLUTIONS, REGION_BOUNDING_BOXES, WEATHER_FILL_METH

MODEL_STRUCTURES = ['grid', 'cluster']
MODEL_TYPES = {
    'mlp': mlp.MutlilayerPerceptron,
    'mean_model': regression_models.MeanModel,
    'large_split': regression_models.LargeSplitModel,
    'cumulative_large_split': regression_models.CumulativeLargeSplitModel,
    'zero_only': regression_models.ZeroModel,
    'zero_inflated_p': zero_inflated_models.ZeroInflatedPoissonRegression,
    'hurdle_p': zero_inflated_models.PoissonHurdleRegression,
    'hurdle_p_floor': zero_inflated_models.PoissonHurdleFloorRegression,
    'hurdle_b': zero_inflated_models.NegativeBinomialHurdleRegression,
    'hurdle_b2': zero_inflated_models.NegativeBinomialHurdleRegression2,
    'neg_binomial': regression_models.NegativeBinomialRegression,
    'logistic': regression_models.LogisticBinaryRegression,
    'poisson': regression_models.PoissonRegression,
    'log_normal': regression_models.LogNormalRegression,
    'linear': regression_models.LinearRegression,
    'persistence_aug': regression_models.PersistenceAugmented,
    'persistence_aug_param': regression_models.PersistenceAugmentedParam,
    'persistence': regression_models.PersistenceModel}
SEPARATED_IGNITIONS = ['unified', 'separated', 'active_only']
MEMORY_TYPES = ['none', 'all', 'decay']
DECAY_METHODS = ['fixed', 'learned']
FORECAST_METHODS = ['separate', 'recursive']
LOG_CORRECTION_METHODS = ['add', 'max']
FIRE_LENGTH_BIAS_THRESH = 5
FILTER_MASKS = ['interior', 'no_ocean']

logger = logging.getLogger('pipeline')


def build_single_model(model_type, covariates, log_covariates, params, response_var='num_det_target',
                       exclude_params=None, t_k=None):
    covs = [x for x in covariates]
    log_covs = [x for x in log_covariates]

    if exclude_params is not None:
        for exc in exclude_params:
            if exc in covs:
                covs.remove(exc)
            if exc in log_covs:
                log_covs.remove(exc)

    model_cls = MODEL_TYPES[model_type]
    model = model_cls(response_var, covs, log_covs, params['log_correction_type'],
                      params['log_correction_constant'], params['regularization_weight'], params['normalize_params'],
                      t_k=t_k,
                      add_exposure=params['exposure'])

    return model


def add_memory_all(X_ds, dates, mem_cov, memory_length, memory_start=1):
    names = []

    values = np.array(X_ds[mem_cov].values)

    # Add autoregressive memory
    for i in range(memory_start, memory_length + 1):
        y_mem = setup_ds.shift_in_time(values, dates, -i, np.zeros)

        name = mem_cov + '_' + str(i)
        names.append(name)

        X_ds.update({name: (('y', 'x', 'time'), y_mem)})

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
        new += X_ds[mem_cov + '_' + str(i + 1)] * vals[i]

    name = mem_cov + '_expon'
    X_ds.update({name: (('y', 'x', 'time'), new)})

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
        added_active = ('num_det' in params['memory_covariates']) or ('num_det' in params['memory_log_covariates'])
        if no_memory or ((active_check_days - 1) > memory_length) or (not added_active):
            memory_start = 1 if (no_memory or (not added_active)) else memory_length + 1
            dates = np.array(list(map(lambda x: pd.Timestamp(x).to_pydatetime().date(), X_ds.time.values)))
            _ = add_memory_all(X_ds, dates, 'num_det', active_check_days - 1, memory_start=memory_start)

        is_active = X_ds.active.values

        for i in range(1, active_check_days):
            vals = X_ds['num_det_' + str(i)].values  # Using memory covariates to avoid recomputing
            is_active = np.logical_or(is_active, vals)

        # name = 'active_' + str(act)
        name = 'active'
        X_ds.update({name: (('y', 'x', 'time'), is_active)})

        # Forward pass
        num_det = X_ds['num_det_target'].values
        large_fire = np.zeros_like(num_det)
        is_cur_active = num_det > 0
        large_fire[:, :, 0] = num_det[:, :, 0]
        for i in range(1, is_active.shape[2]):
            large_fire[:, :, i] = large_fire[:, :, i - 1] * is_cur_active[:, :, i] + num_det[:, :, i]

        # Backward pass
        for i in range(is_active.shape[0] - 2, -1, -1):
            large_fire[:, :, i] = large_fire[:, :, i + 1] * (
                    is_cur_active[:, :, i] & is_cur_active[:, :, i + 1]) + large_fire[:, :, i] * (
                                          is_cur_active[:, :, i] & ~is_cur_active[:, :, i + 1])

        large = large_fire.flatten()
        large = large[large > 0]

        LARGE_FIRE_SIZE = scipy.stats.mstats.mquantiles(large, [params['large_fire_split_percent']])
        print('large', LARGE_FIRE_SIZE)
        large_fire = (large_fire >= LARGE_FIRE_SIZE).astype(np.bool)

        # print('det', X_ds.num_det.values[large_fire])

        name = 'large_fire'
        X_ds.update({name: (('y', 'x', 'time'), large_fire)})


def compute_grad(X_grid_dict):
    for t_k in X_grid_dict:
        X_ds = X_grid_dict[t_k]
        today = X_ds['vpd_%d' % t_k].values
        grad = np.array((X_ds['vpd'] - today) / today)

        grad[np.isnan(grad)] = 0
        grad[grad == np.inf] = 0
        grad[grad == -np.inf] = 0

        grad[grad > 1] = 1

        name = 'vpd_grad'
        X_ds.update({name: (('y', 'x', 'time'), grad)})


def compute_diff(X_grid_dict):
    for X_ds in X_grid_dict.values():
        num_det = X_ds['num_det'].values
        num_det_target = X_ds['num_det_target']

        det_diff = num_det_target - num_det

        name = 'det_diff'
        X_ds.update({name: (('y', 'x', 'time'), det_diff)})


def add_fire_length(X_grid_dict, add_bias):
    for X_ds in X_grid_dict.values():
        covs = []

        active = (X_ds.num_det.values == 1) | (X_ds.num_det_1.values == 1)
        fire_length = np.zeros(active.shape)

        fire_length[:, :, 0] = active[:, :, 0]
        for day in range(1, active.shape[2]):
            fire_length[:, :, day] = (fire_length[:, :, day - 1] + active[:, :, day]) * active[:, :, day]

        name = 'fire_length'
        X_ds.update({name: (('y', 'x', 'time'), fire_length)})

        if add_bias:
            for i in range(1, FIRE_LENGTH_BIAS_THRESH):
                fire_length_bias = fire_length == i

                bias_name = name + '_%d' % i
                covs.append(bias_name)

                X_ds.update({bias_name: (('y', 'x', 'time'), fire_length_bias)})

            # Create bias for all lengths exceeding threshold
            fire_length_bias = fire_length > i

            bias_name = name + '_%d_max' % (i + 1)
            covs.append(bias_name)

            X_ds.update({bias_name: (('y', 'x', 'time'), fire_length_bias)})
        else:
            covs.append(name)

    return covs


def add_ignition_target(X_grid_dict):
    for X_ds in X_grid_dict.values():
        targets = X_ds.num_det_target.values
        ignition = np.zeros(targets.shape, dtype=bool)

        for day in range(1, targets.shape[2]):
            ignition[:, :, day] = (targets[:, :, day] != 0) & (targets[:, :, day - 1] == 0)

        X_ds.update({'ignition': (('y', 'x', 'time'), ignition)})


def add_exposure(X_grid_dict):
    time = len(X_grid_dict[1].time)
    bb = get_default_bounding_box()
    lats, lons = bb.make_grid()
    lats = lats[:, 0]
    widths = 111.321 * np.cos(np.deg2rad(lats - .25)) * .5
    areas_vec = widths * 111 * .5
    areas = np.zeros((33, 55, time))
    areas[:] = areas_vec[:, None, None]

    for X_ds in X_grid_dict.values():
        X_ds.update({'exposure': (('y', 'x', 'time'), areas)})


def add_filter_mask(X_grid_dict, filter_mask, params):
    bb = get_default_bounding_box()
    target_shape = X_grid_dict[1].temperature.values.shape[0:2]

    if filter_mask == 'interior':
        alaska_interior_mask_src = os.path.join('/extra/graffc0/fire_prediction/data',
                                                'processed/masks/alaska_interior_mask_05.nc')
        alaska_interior_mask = xr.open_dataset(alaska_interior_mask_src)

        mask = np.zeros(target_shape, dtype=bool)

        lats, lons = bb.make_grid(inclusive_lon=True)
        for i in range(alaska_interior_mask.mask_Int_05.shape[0]):
            for j in range(alaska_interior_mask.mask_Int_05.shape[1]):
                lat = alaska_interior_mask.Lat_AK_05.values[i, j] + .25
                lon = alaska_interior_mask.Lon_AK_05.values[i, j] - .25

                if alaska_interior_mask.mask_Int_05[i, j] == 1:
                    u = np.where(lats[:, 0] == lat)[0][0]
                    v = np.where(lons[0, :] == lon)[0][0]
                    mask[u, v] = 1

    elif filter_mask == 'no_ocean':
        land_cover_src = os.path.join('/extra/graffc0/fire_prediction/data', 'raw/land_mcd12c1/land_cover.pkl')
        with open(land_cover_src, 'rb') as fin:
            land_cover = pickle.load(fin)

        lc_down = downsample_land_cover(land_cover, target_shape, bb)
        ocean_frac = lc_down[:, :, 0] / 100
        mask = ocean_frac != 1
    else:
        raise ValueError('Invalid value for filter_mask: "%s"' % filter_mask)

    mask_rep = np.broadcast_to(mask[:, :, None], X_grid_dict[1].temperature.values.shape)

    for X_ds in X_grid_dict.values():
        X_ds.update({'filter_mask': (('y', 'x', 'time'), mask_rep)})


def downsample_land_cover(lc, target_shape, target_bb):
    num_classes = np.max(lc) + 1
    land_cover_ds = np.zeros(target_shape + (num_classes,))

    lat_min, lat_max, lon_min, lon_max = target_bb.get()

    ul_lat_ind = int(np.round((90 - lat_max) / .05))
    ul_lon_ind = int(np.round((180 + lon_min) / .05))

    print(ul_lat_ind, ul_lon_ind)

    lc = lc[ul_lat_ind:, ul_lon_ind:]

    LAND_COVER_RES = .05
    TARGET_RES = .5
    res_ratio = int(TARGET_RES / LAND_COVER_RES)
    for i in range(target_shape[0]):
        for j in range(target_shape[1]):
            lc_box = lc[i * res_ratio:(i + 1) * res_ratio, j * res_ratio:(j + 1) * res_ratio]

            unique, counts = np.unique(lc_box, return_counts=True)
            box_counts = dict(zip(unique, counts))

            land_cover_ds[i, j, :] = [box_counts[k] if k in box_counts else 0 for k in range(num_classes)]

    return land_cover_ds


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

    if params['filter_mask'] is not None:
        add_filter_mask(X_grid_dict, params['filter_mask'], params)

    if 'vpd' in params['memory_covariates'] and params['memory_type'] != 'none' and \
            'aug' in params['active_model_type']:
        compute_grad(X_grid_dict)
        compute_diff(X_grid_dict)

    # Ignition target (used for results calculation)
    add_ignition_target(X_grid_dict)

    if 'fire_length' in covariates:
        new_covs = add_fire_length(X_grid_dict, False)
        covariates += new_covs

    if 'fire_length_bias' in covariates:
        covariates.remove('fire_length_bias')
        new_covs = add_fire_length(X_grid_dict, True)
        covariates += new_covs

    if params['exposure']:
        add_exposure(X_grid_dict)

    if params['log_correction_type'] == 'add':
        def log_corr(x):
            return np.log(x + params['log_correction_constant'])
    elif params['log_correction_type'] == 'max':
        def log_corr(x):
            return np.log(np.maximum(x, params['log_correction_constant']))

    """
    for ds in X_grid_dict.values():
        for var in log_covariates:
            ds[var] = log_corr(ds[var])
    """

    return X_grid_dict, covariates, log_covariates


def setup_data(in_files, start_date, end_date, forecast_horizon, parameters):
    # Load data
    X_grid_dict_nw = {k: xr.open_dataset(target) for (k, target) in in_files.items()}

    # Setup data
    years_train = list(range(start_date.year, end_date.year + 1))
    X_grid_dict_nw, covariates, log_covariates = build_covariates(X_grid_dict_nw, parameters)

    logger.debug('Cov.: %s, Log Cov.: %s' % (str(covariates), str(log_covariates)))

    X_grid_dict_nw = {k: filter_fire_season(v, years=years_train) for (k, v) in X_grid_dict_nw.items()}

    # Build y targets
    t_k_arr = list(range(1, forecast_horizon + 1))
    y_grid_dict = setup_ds.build_y_nw(X_grid_dict_nw[1]['num_det'].values, X_grid_dict_nw[1].time.values, t_k_arr,
                                      years_train)

    # Setup data wrappers for corresponding model structures
    if parameters['forecast_method'] == 'recursive':
        all_ds = [X_grid_dict_nw[k] for k in range(1, forecast_horizon + 1)]
        X_grid_dict_nw = {k: mdw.MultidataWrapper(all_ds) for k in X_grid_dict_nw}
    else:
        X_grid_dict_nw = {k: mdw.MultidataWrapper((ds, ds)) for (k, ds) in X_grid_dict_nw.items()}

    return X_grid_dict_nw, y_grid_dict, covariates, log_covariates, years_train


def build_model(covariates, log_covariates, params, t_k):
    """ Select and instantiate model corresponding to params. """

    if params['separated_ignitions'] == 'unified':
        unified_model = build_single_model(params['active_model_type'], covariates, log_covariates, params, t_k=t_k)
        model = grid_models.UnifiedGrid(unified_model)

    elif params['separated_ignitions'] == 'active_only':
        active_model = build_single_model(params['active_model_type'], covariates, log_covariates, params,
                                          exclude_params=params['active_covariates_exclude'], t_k=t_k)
        model = grid_models.ActiveIgnitionGrid(active_model, None)

    elif params['separated_ignitions'] == 'separated':
        active_model = build_single_model(params['active_model_type'], covariates, log_covariates, params,
                                          exclude_params=params['active_covariates_exclude'], t_k=t_k)
        ignition_model = build_single_model(params['ignition_model_type'], covariates, log_covariates, params,
                                            exclude_params=params['ignition_covariates_exclude'], t_k=t_k)
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


def compute_summary_results(results_tr, results_te, X_grid_dict_nw, years, metrics_=None):
    if metrics_ is None:
        metrics_ = [metrics.root_mean_squared_error,
                    metrics.mean_absolute_error]
    summary_results = defaultdict(dict)

    # Compute overall error metrics
    for i, metric in enumerate(metrics_):
        x = ['Avg.'] + list(range(1, len(results_tr) + 1))
        y = list(map(lambda x: metric(*flat(x)), results_tr))
        y = [np.mean(y)] + y
        summary_results['train'][metric.__name__] = (x, y)

        x = ['Avg.'] + list(range(1, len(results_te) + 1))
        y = list(map(lambda x: metric(*flat(x)), results_te))
        y = [np.mean(y)] + y
        summary_results['test'][metric.__name__] = (x, y)

    # ds = filter_fire_season(X_grid_dict_nw[1][0], years=years)
    # active_inds = ds.active.values.flatten()

    # Compute active and igntion error metrics
    if years is not None:
        active_inds = list(map(lambda i: filter_fire_season(X_grid_dict_nw[i][0], years=years).active.values.flatten(),
                               range(1, len(X_grid_dict_nw) + 1)))
        ignition_inds = list(map(lambda x: ~x, active_inds))
    else:
        active_inds = list(map(lambda i: X_grid_dict_nw[i][0].active.values.flatten(),
                               range(1, len(X_grid_dict_nw) + 1)))
        ignition_inds = list(map(lambda x: ~x, active_inds))

    # Active based on day t
    for inds_name, inds in [('active', active_inds), ('ignition', ignition_inds)]:
        for i, metric in enumerate(metrics_):
            x = ['Avg.'] + list(range(1, len(results_te) + 1))
            y = list(map(lambda x: metric(*flat(x[0]), inds=x[1]), zip(results_te, inds)))
            y = [np.mean(y)] + y
            ratio = list(map(lambda x: np.sum(x) / x.size, inds))
            summary_results['test'][metric.__name__ + '_' + inds_name] = (x, y, ratio)

    # Active based on day t+k
    if years is not None:
        active_det_inds = list(
            map(lambda i: filter_fire_season(X_grid_dict_nw[i][0], years=years).active.values.flatten() is True,
                range(1, len(X_grid_dict_nw) + 1)))
        non_zero_target_inds = list(
            map(lambda i: filter_fire_season(X_grid_dict_nw[i][0], years=years).num_det_target.values.flatten() != 0,
                range(1, len(X_grid_dict_nw) + 1)))

        active_inds = list(map(lambda x: (x[0] & x[1]), zip(active_det_inds, non_zero_target_inds)))
        extinction_inds = list(map(lambda x: (x[0] & ~x[1]), zip(active_det_inds, non_zero_target_inds)))
        zero_zero_inds = list(map(lambda x: (~x[0] & ~x[1]), zip(active_det_inds, non_zero_target_inds)))
        ignition_inds = list(map(lambda x: (~x[0] & x[1]), zip(active_det_inds, non_zero_target_inds)))

    else:
        active_det_inds = list(map(lambda i: X_grid_dict_nw[i][0].active.values.flatten() is True,
                                   range(1, len(X_grid_dict_nw) + 1)))
        non_zero_target_inds = list(map(lambda i: X_grid_dict_nw[i][0].num_det_target.values.flatten() != 0,
                                        range(1, len(X_grid_dict_nw) + 1)))

        active_inds = list(map(lambda x: (x[0] & x[1]), zip(active_det_inds, non_zero_target_inds)))
        extinction_inds = list(map(lambda x: (x[0] & ~x[1]), zip(active_det_inds, non_zero_target_inds)))
        zero_zero_inds = list(map(lambda x: (~x[0] & ~x[1]), zip(active_det_inds, non_zero_target_inds)))
        ignition_inds = list(map(lambda x: (~x[0] & x[1]), zip(active_det_inds, non_zero_target_inds)))

    # print('Zero-Zero Mismatch Predict', np.sum(results_te[0][1].flatten()[zero_zero_inds[0]]!=0))
    # print('Zero-Zero Mismatch Test', np.sum(results_te[0][0].flatten()[zero_zero_inds[0]]!=0))

    # zero_target_inds = list(map(lambda i: filter_fire_season(X_grid_dict_nw[i][0],
    # years=years).num_det_target.values.flatten()==0, range(1,len(X_grid_dict_nw)+1))) print('Zero-Zero Mismatch
    # Test', np.sum(results_te[0][0].flatten()!=filter_fire_season(X_grid_dict_nw[i][0],
    # years=years).num_det_target.values.flatten()))

    for inds_name, inds in [('active_target', active_inds), ('ignition_target', ignition_inds),
                            ('zero_zero_target', zero_zero_inds), ('extinction_target', extinction_inds)]:
        for i, metric in enumerate(metrics_):
            x = ['Avg.'] + list(range(1, len(results_te) + 1))
            y = list(map(lambda x: metric(*flat(x[0]), inds=x[1]), zip(results_te, inds)))
            y = [np.mean(y)] + y
            ratio = list(map(lambda x: np.sum(x) / x.size, inds))
            summary_results['test'][metric.__name__ + '_' + inds_name] = (x, y, ratio)

    return summary_results


class TrainModel(luigi.Task):
    data_dir = luigi.parameter.Parameter()
    experiment_dir = luigi.parameter.Parameter()

    start_date = luigi.parameter.DateParameter(default=dt.date(2007, 1, 1))
    end_date = luigi.parameter.DateParameter(default=dt.date(2016, 12, 31))

    resolution = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS, default='4')
    bounding_box_name = luigi.parameter.ChoiceParameter(choices=REGION_BOUNDING_BOXES.keys(), default='alaska')

    fire_season_start = luigi.parameter.DateParameter(default=dt.date(2007, 5, 14))
    fire_season_end = luigi.parameter.DateParameter(default=dt.date(2007, 8, 31))

    model_structure = luigi.parameter.ChoiceParameter(choices=MODEL_STRUCTURES)
    separated_ignitions = luigi.parameter.ChoiceParameter(choices=SEPARATED_IGNITIONS)
    active_model_type = luigi.parameter.ChoiceParameter(choices=list(MODEL_TYPES.keys()))
    ignition_model_type = luigi.parameter.ChoiceParameter(choices=list(MODEL_TYPES.keys()), default=None)
    covariates = luigi.parameter.ListParameter()
    ignition_covariates_exclude = luigi.parameter.ListParameter()
    active_covariates_exclude = luigi.parameter.ListParameter()
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
    filter_mask = luigi.parameter.ChoiceParameter(choices=FILTER_MASKS)
    large_fire_split_percent = luigi.parameter.NumericalParameter(var_type=float, min_value=0, max_value=1, default=.9)

    forecast_horizon = luigi.parameter.NumericalParameter(var_type=int, min_value=1, max_value=10, default=5)
    rain_offset = luigi.parameter.NumericalParameter(var_type=int, min_value=-24, max_value=24, default=0)
    years_test = luigi.parameter.ListParameter(default=None)

    use_era = luigi.parameter.BoolParameter(default=False)
    exposure = luigi.parameter.BoolParameter(default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_k_arr = None
        self.train_parameters = None
        self.job_id = None

    def requires(self):
        self.t_k_arr = range(1, self.forecast_horizon + 1)
        if self.model_structure == 'grid':
            tasks = {
                k: GridDatasetGeneration(data_dir=self.data_dir, start_date=self.start_date, end_date=self.end_date,
                                         resolution=self.resolution, bounding_box_name=self.bounding_box_name,
                                         fill_method=self.fill_method,
                                         forecast_horizon=k, rain_offset=self.rain_offset, use_era=self.use_era) for k
                in
                self.t_k_arr}
        else:
            raise NotImplementedError('Training cluster models not supported yet')

        return tasks

    def run(self):
        # Setup data
        X_grid_dict_nw, y_grid_dict, covariates, log_covariates, years_train = setup_data(
            {k: v.path for (k, v) in self.input().items()}, self.start_date, self.end_date, self.forecast_horizon,
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

        X_ds = X_grid_dict_nw[1][0]
        summary_results = compute_summary_results(results[0], results[1], X_grid_dict_nw, years_test)

        out_dict = {'models': models, 'summary_results': summary_results, 'params': self.train_parameters}

        # Save model and parameters
        with self.output().temporary_path() as temp_output_path:
            with open(temp_output_path, 'wb') as fout:
                pickle.dump(out_dict, fout)

        logger.info('JOB ID: %d -- %s' % (self.job_id, str(self.train_parameters)))
        logger.debug('RESULTS: %s -- %s' % (str(summary_results), str(self.train_parameters)))

    def output(self):
        self.train_parameters = {
            'model_structure': self.model_structure,
            'separated_ignitions': self.separated_ignitions,
            'active_model_type': self.active_model_type,
            'ignition_model_type': self.ignition_model_type,
            'covariates': self.covariates,
            'ignition_covariates_exclude': self.ignition_covariates_exclude,
            'active_covariates_exclude': self.active_covariates_exclude,
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
            'exposure': self.exposure,
            'normalize_params': self.normalize_params,
            'filter_mask': self.filter_mask,
            'large_fire_split_percent': self.large_fire_split_percent}

        # fn = '_'.join(list(map(str, self.train_parameters))) + '.pkl'
        self.job_id = create_job_id(self.train_parameters)
        fn = str(self.job_id) + '.pkl'

        dest_path = os.path.join(self.experiment_dir, fn)

        return luigi.LocalTarget(dest_path)
