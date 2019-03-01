import datetime as dt
import luigi
import os
import logging

from pipeline.train_pipeline import TrainModel

logger = logging.getLogger('pipeline')
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

start_date = dt.date(2007, 1, 1)
end_date = dt.date(2016, 12, 31)
data_dir = '/lv_scratch/scratch/graffc0/fire_prediction/data/'
experiment_dir = '/lv_scratch/scratch/graffc0/fire_prediction/experiments/'

WEATHER_ALL = ['temperature', 'humidity', 'rain', 'wind']
WEATHER_ALL_VPD = ['temperature', 'humidity', 'rain', 'wind', 'vpd']
WEATHER_ALL_NO_WIND = ['temperature', 'humidity', 'rain']
WEATHER_ALL_NO_RAIN = ['temperature', 'humidity', 'wind']
TEMP_HUMID = ['temperature', 'humidity']
DETECTIONS = ['num_det']
FIRE_LENGTH = ['fire_length']
FIRE_LENGTH_BIAS = ['fire_length_bias']
ALL_YEARS = None
IN_SAMPLE = [None]

#'decay_values': {'default': .05, 'num_det': .05, 'temperature': .25, 'humidity': .5, 'rain': .25},
params = {
        'model_structure': 'grid',
        'separated_ignitions': 'active_only',
        'active_model_type': 'poisson',
        'ignition_model_type':'poisson',
        'covariates': ['temperature'],
        'ignition_covariates_exclude': [],
        'log_covariates': DETECTIONS,
        'memory_type': 'none',
        'memory_covariates': WEATHER_ALL,
        'memory_log_covariates': DETECTIONS,
        'memory_length': 10,
        'decay_method': 'fixed',
        'decay_values': {'default': .05, 'num_det': .05, 'temperature': .25, 'humidity': .5, 'rain': .25},
        'forecast_method': 'separate',
        'active_check_days': 2,
        'regularization_weight': None,
        'log_correction_type': 'max',
        'log_correction_constant': .5,
        'fill_method': 'interpolate',
        'forecast_horizon': 5,
        'years_test': ALL_YEARS,
        'normalize_params': False,
        'rain_offset': 0,
        'exposure': False,
        'use_era': True,
        'large_fire_split_percent': .9,
}

tasks = []
new_task = TrainModel(data_dir=data_dir, experiment_dir=experiment_dir, start_date=start_date, end_date=end_date, 
        resolution='4', bounding_box_name='alaska', **params)

tasks.append(new_task)

#luigi.build(tasks, local_scheduler=False, worker_scheduler_factory=None, workers=5, scheduler_port=8881,
#    log_level='INFO')
luigi.build(tasks, local_scheduler=True, worker_scheduler_factory=None, workers=1, scheduler_port=8881,
        log_level='INFO')
