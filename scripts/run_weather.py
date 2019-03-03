import datetime as dt
import luigi
import logging

from src.pipeline.weather_pipeline import WeatherFillMissingValues, WeatherGridGeneration

data_dir = '/extra/graffc0/fire_prediction/data/'

logger = logging.getLogger('pipeline')
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

FILL_MISSING = False

if FILL_MISSING:
    start_date = dt.date(2007, 1, 1)
    end_date = dt.date(2016, 12, 31)
    luigi.build([WeatherFillMissingValues(data_dir=data_dir, start_date=start_date, end_date=end_date,
                                          resolution='4', bounding_box_name='alaska', fill_method='integrate')],
                local_scheduler=False, worker_scheduler_factory=None, workers=1, scheduler_port=8881, log_level='INFO')
else:
    start_date = dt.date(2007, 1, 1)
    end_date = dt.date(2016, 12, 31)
    luigi.build([WeatherGridGeneration(data_dir=data_dir, start_date=start_date, end_date=end_date,
                                       resolution='4', bounding_box_name='alaska', fill_method='integrate')],
                local_scheduler=False, worker_scheduler_factory=None, workers=2, scheduler_port=8881, log_level='INFO')
