"""
Creates the project workflow pipeline from data ingestion to model execution.
"""

import luigi
import luigi.contrib.ftp
import os
import h5py
import pygrib
import pickle
import datetime as dt
from ftplib import FTP
import logging
import numpy as np
import pandas as pd
import xarray as xr

from io import BytesIO
import pytz

from time import time

from data import grib

from data.gfs_choices import GFS_BOUNDING_BOXES, GFS_MEASUREMENT_SEL
import parse # For latlon param type
from helper.geometry import LatLonBoundingBox # For latlon param type

GFS_SERVER_NAME = 'nomads.ncdc.noaa.gov'  # Server from which to pull the GFS data
GFS_SERVER_USERNAME = 'anonymous'
GFS_SERVER_PASSWORD = 'graffc@uci.edu'

GFS_SERVER_DATA_DIR = "GFS/analysis_only/"  # location on server of GFS data

GFS_START_YEAR = 2007
GFS_END_YEAR = 2016

year_month_dir_fmt = "%d%.2d"
year_month_day_dir_fmt = "%d%.2d%.2d"
grib_file_fmt_half_deg = "gfsanl_4_%s_%.2d00_%.3d.grb2"
grib_file_fmt_one_deg = "gfsanl_3_%s_%.2d00_%.3d.grb"

GFS_TIMES = [0, 6, 12, 18]
GFS_OFFSETS = [0, 3, 6]
GFS_RESOLUTIONS = ['3', '4']

def_name_conversion_dict = {'Surface air relative humidity': 'humidity', '2 metre relative humidity': 'humidity',
        'Relative humidity': 'humidity', '10 metre U wind component': 'U wind component', 
        '10 metre V wind component': 'V wind component', 'Convective available potential energy': 'cape', 
        'Planetary boundary layer height': 'pbl_height', 'Volumetric soil moisture content': 'soil_moisture' }

class TimeTaskMixin(object):
    '''
    A mixin that when added to a luigi task, will print out
    the tasks execution time to standard out, when the task is
    finished
    '''
    @luigi.Task.event_handler(luigi.Event.PROCESSING_TIME)
    def print_execution_time(self, processing_time):
        print('=== PROCESSING TIME === ' + str(processing_time))

def build_gfs_server_file_path(data_dir, resolution, date, time=None, offset=None):
    year_month = year_month_dir_fmt % (date.year, date.month)
    year_month_day = year_month_day = year_month_day_dir_fmt % (date.year, date.month, date.day)

    # If time and offset are not specified, return day directory
    if time is None or offset is None:
        return os.path.join(data_dir, year_month, year_month_day)

    # Otherwise, build full file path
    if resolution == '3':
        grib_file = grib_file_fmt_one_deg % (year_month_day, time, offset)
    elif resolution == '4':
        grib_file = grib_file_fmt_half_deg % (year_month_day, time, offset)
    else:
        raise ValueError('"%s" is not a valid resolution.' % resolution)

    return os.path.join(data_dir, year_month, year_month_day, grib_file)

def change_data_dir_path(server_data_dir, local_data_dir, path):
    path = path.split(server_data_dir)[1]
    return os.path.join(local_data_dir, path.lstrip('/'))

class FtpFile(luigi.ExternalTask):
    server_name = luigi.parameter.Parameter(default=GFS_SERVER_NAME)
    server_username = luigi.parameter.Parameter(default=GFS_SERVER_USERNAME, significant=False)
    server_password = luigi.parameter.Parameter(default=GFS_SERVER_PASSWORD, significant=False)

    file_path = luigi.parameter.Parameter()

    resources = {'ftp': 1}

    def output(self):
        return luigi.contrib.ftp.RemoteTarget(self.file_path, host=self.server_name, username=self.server_username,
                password=self.server_password)

class GfsFtpFileDownload(luigi.Task):
    server_name = luigi.parameter.Parameter(default=GFS_SERVER_NAME)
    server_username = luigi.parameter.Parameter(default=GFS_SERVER_USERNAME, significant=False)
    server_password = luigi.parameter.Parameter(default=GFS_SERVER_PASSWORD, significant=False)
    server_data_dir = luigi.parameter.Parameter(default=GFS_SERVER_DATA_DIR)
    
    local_data_dir = luigi.parameter.Parameter()

    resolution_sel = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS)
    date_sel = luigi.parameter.DateParameter()
    time_sel = luigi.parameter.ChoiceParameter(choices=GFS_TIMES, var_type=int)
    offset_sel = luigi.parameter.ChoiceParameter(choices=GFS_OFFSETS, var_type=int)

    resources = {'ftp': 1}

    def requires(self):
        file_path = build_gfs_server_file_path(self.server_data_dir, self.resolution_sel, self.date_sel, 
                self.time_sel, self.offset_sel)
        return FtpFile(file_path=file_path)

    def run(self):
        # Copy ftp file from server to local dest
        with self.output().temporary_path() as temp_output_path:
            self.input().get(temp_output_path)

    def output(self):
        dest_path = change_data_dir_path(self.server_data_dir, self.local_data_dir, self.input().path)
        return luigi.LocalTarget(dest_path)

class GfsFilterMeasurements(luigi.Task):
    src_data_dir = luigi.parameter.Parameter() # Can we remove this param by storing info in the LocalTarget input
    dest_data_dir = luigi.parameter.Parameter()

    resolution_sel = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS)
    date_sel = luigi.parameter.DateParameter()
    time_sel = luigi.parameter.ChoiceParameter(choices=GFS_TIMES, var_type=int)
    offset_sel = luigi.parameter.ChoiceParameter(choices=GFS_OFFSETS, var_type=int)

    measurement_sel_name = luigi.parameter.ChoiceParameter(choices=GFS_MEASUREMENT_SEL.keys())
    bounding_box_sel_name = luigi.parameter.ChoiceParameter(choices=GFS_BOUNDING_BOXES.keys())

    def requires(self):
        return GfsFtpFileDownload(local_data_dir=self.src_data_dir, resolution_sel=self.resolution_sel,
                date_sel=self.date_sel, time_sel=self.time_sel, offset_sel=self.offset_sel)

    def run(self):
        # Read grib file and extract selected measurements
        with pygrib.open(self.input().path) as fin:
            selections = GFS_MEASUREMENT_SEL[self.measurement_sel_name]
            bounding_box = GFS_BOUNDING_BOXES[self.bounding_box_sel_name]

            extracted = grib.GribSelector(selections, bounding_box).select(fin)

        # Write extracted measurements as hdf5
        with self.output().temporary_path() as temp_output_path:
            f = h5py.File(temp_output_path, 'w')

            # Write each measurement as a separate dataset
            for k,v in extracted.items():
                dset = f.create_dataset(k, data=v['values'], compression='lzf')
                dset.attrs['units'] = v['units']

            # Set bouding box for entire file (all datasets share the same bounding box)
            f.attrs['bounding_box'] = v['bounding_box'].get()


    def output(self):
        dest_path = change_data_dir_path(self.src_data_dir, self.dest_data_dir, self.input().path)
        dest_path, _ = os.path.splitext(dest_path)
        dest_path += '_%s_%s' % (self.measurement_sel_name, self.bounding_box_sel_name)
        dest_path += '.hdf5'
        return luigi.LocalTarget(dest_path)

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + dt.timedelta(n)

def ftp_file_exists(ftp, path):
    dirname, file_name = os.path.split(path)

    return path in files or file_name in files

class GfsGetAvailableFilesList(luigi.Task):
    server_name = luigi.parameter.Parameter(default=GFS_SERVER_NAME)
    server_username = luigi.parameter.Parameter(default=GFS_SERVER_USERNAME, significant=False)
    server_password = luigi.parameter.Parameter(default=GFS_SERVER_PASSWORD, significant=False)
    server_data_dir = luigi.parameter.Parameter(default=GFS_SERVER_DATA_DIR)

    local_data_dir = luigi.parameter.Parameter()

    resolution_sel = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS)
    start_date_sel = luigi.parameter.DateParameter()
    end_date_sel = luigi.parameter.DateParameter()

    resources = {'ftp': 1}

    def run(self):
        ftp = FTP(self.server_name)
        ftp.login(self.server_username, self.server_password)

        available_files = []

        for date in daterange(self.start_date_sel, self.end_date_sel + dt.timedelta(1)):
            day_dir_path = build_gfs_server_file_path(self.server_data_dir, self.resolution_sel, date)
            files = ftp.nlst(day_dir_path)

            for time,offset in [(t,o) for t in GFS_TIMES for o in GFS_OFFSETS]:
                file_path = build_gfs_server_file_path(self.server_data_dir, self.resolution_sel, date, time, offset)
                dirname, file_name = os.path.split(file_path)

                if file_path in files or file_name in files: 
                    available_files.append(file_path)
                else:
                    logging.debug('Missing File: resolution %s year %d month %d day %d time %d offset %d not on server'
                            % (self.resolution_sel, date.year, date.month, date.day, time, offset))

        # Clean-up
        ftp.quit()

        with self.output().temporary_path() as temp_output_path:
            with open(temp_output_path, 'wb') as fout:
                pickle.dump(available_files, fout, protocol=pickle.HIGHEST_PROTOCOL)

    def output(self):
        date_fmt = '%m%d%Y'

        start_date_str = self.start_date_sel.strftime(date_fmt)
        end_date_str =  self.end_date_sel.strftime(date_fmt)

        file_name = 'gfsanl_%s_available_%s-%s.pkl' % (self.resolution_sel, start_date_str, end_date_str)
        dest_path = os.path.join(self.local_data_dir, file_name)

        return luigi.LocalTarget(dest_path)

def create_filter_task_from_file_name(fn, src_data_dir, dest_data_dir, measurement_sel_name, bounding_box_sel_name):
    fmt = 'gfsanl_{resolution}_{date}_{time:d}_{offset:d}.{}' 
    p = parse.parse(fmt, fn)

    year,month,day = int(p['date'][:4]), int(p['date'][4:6]), int(p['date'][6:8])

    resolution_sel = p['resolution']
    date_sel = dt.date(year, month, day)
    time_sel=p['time']//100
    offset_sel=p['offset']

    task = GfsFilterMeasurements(src_data_dir=src_data_dir, dest_data_dir=dest_data_dir, resolution_sel=resolution_sel,
            date_sel=date_sel, time_sel=time_sel, offset_sel=offset_sel,measurement_sel_name=measurement_sel_name, 
            bounding_box_sel_name=bounding_box_sel_name)
    
    return task

def create_true_dates(start_date, end_date, times, offsets):
    dates = list(daterange(start_date, end_date + dt.timedelta(1)))
    times = list(map(lambda x: dt.time(x), times))
    offsets = list(map(lambda x: dt.timedelta(hours=x), offsets))

    true_dates, true_offsets = zip(*[(dt.datetime.combine(d, t), o) for d in dates for t in times for o in offsets])

    true_dates = pd.to_datetime(true_dates)
    return true_dates, np.array(true_offsets)

def get_date_ind(true_dates, true_offsets, date_ind, datetime_sel, offset_sel):
    while True:
        if (true_dates[date_ind] == datetime_sel) and (true_offsets[date_ind]==offset_sel):
            return date_ind
        date_ind += 1
    
    raise ValueError('Unable to find matching date and offset for "%s", "%s".' % (str(datetime_sel), str(offset_sel)))

class GfsAggregate(luigi.Task):
    raw_data_dir = luigi.parameter.Parameter()
    src_data_dir = luigi.parameter.Parameter()
    dest_data_dir = luigi.parameter.Parameter()

    resolution_sel = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS)
    start_date_sel = luigi.parameter.DateParameter()
    end_date_sel = luigi.parameter.DateParameter()

    measurement_sel_name = luigi.parameter.ChoiceParameter(choices=GFS_MEASUREMENT_SEL.keys(), default='default_v1')
    bounding_box_sel_name = luigi.parameter.ChoiceParameter(choices=GFS_BOUNDING_BOXES.keys(), default='global')

    def requires(self):
        return GfsGetAvailableFilesList(local_data_dir=self.raw_data_dir, resolution_sel=self.resolution_sel,
                start_date_sel=self.start_date_sel, end_date_sel=self.end_date_sel)

    def run(self):
        with open(self.input().path, 'rb') as fin:
            available_files = pickle.load(fin)

        available_file_names = map(lambda x: os.path.split(x)[1], available_files)
        required_tasks = list(map(lambda x: create_filter_task_from_file_name(x, self.raw_data_dir, self.src_data_dir, 
            self.measurement_sel_name, self.bounding_box_sel_name), available_file_names))

        # Dynamic requirements on filtering all of the available files
        yield required_tasks

        # Create dates and lat/lon for xarray coords
        true_dates, true_offsets = create_true_dates(self.start_date_sel, self.end_date_sel, GFS_TIMES, GFS_OFFSETS)

        grid_increment = 1. if self.resolution_sel == '3' else .5
        lats, lons = GFS_BOUNDING_BOXES[self.bounding_box_sel_name].make_grid(grid_increment, grid_increment)
        lats, lons = lats[:,0], lons[0,:]

        # Add empty array for each measurements
        data_arrays = {}
        for s in GFS_MEASUREMENT_SEL[self.measurement_sel_name]:
            data = np.empty((len(lats), len(lons), len(true_dates)), dtype=s.dtype)
            data.fill(np.nan)

            data_arrays[s.name] = (['y', 'x', 'time'], data)

        ds = xr.Dataset(data_arrays, coords={'lat': (['y'], lats), 'lon': (['x'], lons), 'time': true_dates,
            'offset': (['time'], true_offsets)})

        # For each file (one per task), add the single day slice to the dataset
        date_ind = 0
        for task in required_tasks:
            file_path = task.output().path
            date_sel, time_sel, offset_sel  = task.date_sel, task.time_sel, task.offset_sel

            datetime_sel = dt.datetime.combine(date_sel, dt.time(time_sel))
            offset_sel = dt.timedelta(hours=offset_sel)

            date_ind = get_date_ind(true_dates, true_offsets, date_ind, datetime_sel, offset_sel)

            with h5py.File(file_path, 'r') as fin:
                for measurement in fin:
                    ds[measurement][:,:,date_ind] = fin[measurement][:]
                    ds[measurement].attrs['units'] = fin[measurement].attrs['units']

            date_ind += 1

        # Write output
        with self.output().temporary_path() as temp_output_path:
            encoding = {s.name: {'zlib': True, 'complevel': 1}  for s in GFS_MEASUREMENT_SEL[self.measurement_sel_name]}
            ds.to_netcdf(temp_output_path, encoding=encoding)

    def output(self):
        date_fmt = '%m%d%Y'

        start_date_str = self.start_date_sel.strftime(date_fmt)
        end_date_str =  self.end_date_sel.strftime(date_fmt)

        file_name = 'gfsanl_%s_%s_%s_%s_%s.nc' % (self.resolution_sel, self.measurement_sel_name, 
                self.bounding_box_sel_name, start_date_str, end_date_str)

        dest_path = os.path.join(self.dest_data_dir, file_name)

        return luigi.LocalTarget(dest_path)

class GfsFilterRegion(luigi.Task):
    raw_data_dir = luigi.parameter.Parameter()
    interim_filtered_data_dir = luigi.parameter.Parameter()
    interim_aggregated_data_dir = luigi.parameter.Parameter()
    dest_data_dir = luigi.parameter.Parameter()

    resolution_sel = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS)

    start_date_sel = luigi.parameter.DateParameter()
    end_date_sel = luigi.parameter.DateParameter()

    bounding_box_sel_name = luigi.parameter.ChoiceParameter(choices=GFS_BOUNDING_BOXES.keys())

    def requires(self):
        return GfsAggregate(raw_data_dir=self.raw_data_dir, src_data_dir=self.interim_filtered_data_dir,
                dest_data_dir=self.interim_aggregated_data_dir, resolution_sel=self.resolution_sel, 
                start_date_sel=self.start_date_sel, end_date_sel=self.end_date_sel)

    def run(self):
        bounding_box = GFS_BOUNDING_BOXES[self.bounding_box_sel_name]
        lat_min, lat_max, lon_min, lon_max = bounding_box.get()

        start_date, end_date = np.datetime64(self.start_date_sel), np.datetime64(self.end_date_sel)
        end_date += np.timedelta64(1, 'D')

        # Select lat/lon and date range
        with xr.open_dataset(self.input().path) as ds:
            time_ind = (ds.time >= start_date) & (ds.time < end_date)
            lat_ind = (ds.lat >= lat_min) & (ds.lat <= lat_max)
            lon_ind = (ds.lon >= lon_min) & (ds.lon <= lon_max)

            ds_sel = ds[{'y': lat_ind, 'x': lon_ind, 'time': time_ind}]

        # Save selected ranges
        with self.output().temporary_path() as temp_output_path:
            ds_sel.to_netcdf(temp_output_path)

    def output(self):
        dirname, fn = os.path.split(self.input().path)
        fn, _ = os.path.splitext(fn)
        fn = fn + '_%s' % self.bounding_box_sel_name + '.nc'
        dest_path = os.path.join(self.dest_data_dir, fn)

        return luigi.LocalTarget(dest_path)
