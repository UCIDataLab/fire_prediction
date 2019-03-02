import datetime as dt
import luigi
import os

from modis_pipeline import ModisFilterRegion
from fire_pipeline import FireGridGeneration

"""
start_date = dt.date(2007, 1, 1)
end_date = dt.date(2016, 12, 31)
data_dir = '/extra/graffc0/fire_prediction/data/'
luigi.build([ModisFilterRegion(data_dir=data_dir, start_month_sel=start_date, end_month_sel=end_date, 
    bounding_box_sel_name='alaska')], local_scheduler=False, worker_scheduler_factory=None, workers=1, 
    scheduler_port=8881, log_level='INFO')
"""

start_date = dt.date(2007, 1, 1)
end_date = dt.date(2016, 12, 31)
data_dir = '/extra/graffc0/fire_prediction/data/'
luigi.build([FireGridGeneration(data_dir=data_dir, start_date=start_date, end_date=end_date,
                                bounding_box_sel_name='alaska')], local_scheduler=False, worker_scheduler_factory=None,
            workers=1,
            scheduler_port=8881, log_level='INFO')
