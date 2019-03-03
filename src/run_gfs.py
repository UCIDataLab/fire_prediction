import datetime as dt
import logging

import luigi

from src.pipeline.gfs_pipeline import GfsFilterRegion

# date = dt.date(2007, 1, 1)
# luigi.build([GfsFileFilterMeasurements(src_data_dir='./temp', dest_data_dir='./temp2',  resolution_sel='4',
#    date_sel=date, time_sel=0, offset_sel=0)], local_scheduler=True, worker_scheduler_factory=None)

# start_date = dt.date(2007, 1, 1)
# end_date = dt.date(2007, 1, 7)
# luigi.build([GfsGetAvailableFilesList(local_data_dir='./temp', resolution_sel='4', start_date_sel=start_date,
#    end_date_sel=end_date)], local_scheduler=True, worker_scheduler_factory=None)

# start_date = dt.date(2007, 1, 1) end_date = dt.date(2007, 1, 2) luigi.build([GfsAggregate(raw_data_dir='./temp',
# src_data_dir='./temp2', dest_data_dir='./temp3', resolution_sel='4', start_date_sel=start_date,
# end_date_sel=end_date)], local_scheduler=False, worker_scheduler_factory=None, workers=4)


logging.basicConfig(level=logging.DEBUG)

data_dir = '/extra/graffc0/fire_prediction/data/'

"""
start_date = dt.date(2007, 1, 1)
end_date = dt.date(2016, 12, 31)
data_dir = '/extra/graffc0/fire_prediction/data/'
luigi.build([GfsFilterRegion(raw_data_dir=os.path.join(data_dir, 'raw/gfs/3/'),
                             interim_filtered_data_dir=os.path.join(data_dir,'interim/gfs/filtered/3/'),
                             interim_aggregated_data_dir=os.path.join(data_dir, 'interim/gfs/aggregated/3'),
                             dest_data_dir=os.path.join(data_dir,'interim/gfs/region/3'), resolution_sel='3',
                             start_date_sel=start_date, end_date_sel=end_date, bounding_box_sel_name='alaska')],
            local_scheduler=False, worker_scheduler_factory=None, workers=1, scheduler_port=8881, log_level='INFO')
"""

start_date = dt.date(2007, 1, 1)
end_date = dt.date(2016, 12, 31)
luigi.build([GfsFilterRegion(data_dir=data_dir, start_date=start_date, end_date=end_date,
                             bounding_box_name='alaska', resolution='4')], local_scheduler=False,
            worker_scheduler_factory=None, workers=3,
            scheduler_port=8881, log_level='INFO')
