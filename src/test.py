import datetime as dt
import luigi

from pipeline import GfsFtpFileDownload, GfsFilterMeasurements, GfsGetAvailableFilesList, GfsAggregate, GfsFilterRegion

#date = dt.date(2007, 1, 1)
#luigi.build([GfsFileFilterMeasurements(src_data_dir='./temp', dest_data_dir='./temp2',  resolution_sel='4',
#    date_sel=date, time_sel=0, offset_sel=0)], local_scheduler=True, worker_scheduler_factory=None)

#start_date = dt.date(2007, 1, 1)
#end_date = dt.date(2007, 1, 7)
#luigi.build([GfsGetAvailableFilesList(local_data_dir='./temp', resolution_sel='4', start_date_sel=start_date,
#    end_date_sel=end_date)], local_scheduler=True, worker_scheduler_factory=None)

#start_date = dt.date(2007, 1, 1)
#end_date = dt.date(2007, 1, 2)
#luigi.build([GfsAggregate(raw_data_dir='./temp', src_data_dir='./temp2', dest_data_dir='./temp3', resolution_sel='4',
#    start_date_sel=start_date, end_date_sel=end_date)], local_scheduler=False, worker_scheduler_factory=None, workers=4)

start_date = dt.date(2007, 1, 1)
end_date = dt.date(2007, 1, 2)
luigi.build([GfsFilterRegion(raw_data_dir='./test/raw', interim_filtered_data_dir='./test/interim/filtered',
    interim_aggregated_data_dir='./test/interim/aggregated', dest_data_dir='./test/interim/region', resolution_sel='4',
    start_date_sel=start_date, end_date_sel=end_date, bounding_box_sel_name='alaska')],
    local_scheduler=False, worker_scheduler_factory=None, workers=1)
