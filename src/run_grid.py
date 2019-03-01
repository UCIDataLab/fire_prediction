import datetime as dt
import luigi
import os
import logging

from pipeline.dataset_pipeline import GridDatasetGeneration

logger = logging.getLogger('pipeline')
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

start_date = dt.date(2007, 1, 1)
end_date = dt.date(2016, 12, 31)
#data_dir = '/extra/graffc0/fire_prediction/data/'
data_dir = '/lv_scratch/scratch/graffc0/fire_prediction/data/'
#tasks = [GridDatasetGeneration(data_dir=data_dir, start_date=start_date, end_date=end_date, 
#    resolution='4', bounding_box_name='alaska', fill_method='integrate_interp', forecast_horizon=fh, rain_offset=0) 
#    for fh in [1,2,3,4,5]]

tasks = [GridDatasetGeneration(data_dir=data_dir, start_date=start_date, end_date=end_date, 
    resolution='4', bounding_box_name='alaska', fill_method='integrate_interp', forecast_horizon=1, rain_offset=0)]

luigi.build(tasks, local_scheduler=True, worker_scheduler_factory=None, workers=1, scheduler_port=8881,
    log_level='DEBUG')
