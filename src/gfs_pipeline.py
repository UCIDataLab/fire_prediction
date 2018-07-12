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
import tempfile
import shutil

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

logger = logging.getLogger('pipeline')

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
            with h5py.File(temp_output_path, 'w') as f:
                # Write each measurement as a separate dataset
                for k,v in extracted.items():
                    dset = f.create_dataset(k, data=v['values'], compression='lzf')
                    dset.attrs['units'] = v['units']

                # Set bouding box for entire file (all datasets share the same bounding box)
                if len(extracted) > 0:
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
            try:
                files = ftp.nlst(day_dir_path)
            except:
                logger.debug('Misssing Day: resolution %s year %d month %d day %d' % (self.resolution_sel, date.year, date.month, date.day))
                continue

            for time,offset in [(t,o) for t in GFS_TIMES for o in GFS_OFFSETS]:
                file_path = build_gfs_server_file_path(self.server_data_dir, self.resolution_sel, date, time, offset)
                dirname, file_name = os.path.split(file_path)

                if file_path in files or file_name in files: 
                    available_files.append(file_path)
                else:
                    logger.debug('Missing File: resolution %s year %d month %d day %d time %d offset %d not on server'
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

def create_filter_task_from_file_name(fn, src_data_dir, dest_data_dir, measurement_sel_name, bounding_box_sel_name, 
        year_sel):
    fmt = 'gfsanl_{resolution}_{date}_{time:d}_{offset:d}.{}' 
    p = parse.parse(fmt, fn)

    year,month,day = int(p['date'][:4]), int(p['date'][4:6]), int(p['date'][6:8])

    if year != year_sel:
        return None

    resolution_sel = p['resolution']
    date_sel = dt.date(year, month, day)
    time_sel=p['time']//100
    offset_sel=p['offset']

    task = GfsFilterMeasurements(src_data_dir=src_data_dir, dest_data_dir=dest_data_dir, resolution_sel=resolution_sel,
            date_sel=date_sel, time_sel=time_sel, offset_sel=offset_sel,measurement_sel_name=measurement_sel_name, 
            bounding_box_sel_name=bounding_box_sel_name)
    
    return task

def create_true_dates(start_date, end_date, times, offsets):
    """ Inclusive of dates. """
    dates = list(daterange(start_date, end_date + dt.timedelta(1)))
    times = list(map(lambda x: dt.time(x), times))
    offsets = list(map(lambda x: dt.timedelta(hours=x), offsets))

    true_dates, true_offsets = zip(*[(dt.datetime.combine(d, t), o) for d in dates for t in times for o in offsets])

    true_dates = pd.to_datetime(true_dates)
    return true_dates, np.array(true_offsets)

def get_date_ind(true_dates, true_offsets, date_ind, datetime_sel, offset_sel):
    skipped = []
    while True:
        if (true_dates[date_ind] == datetime_sel) and (true_offsets[date_ind]==offset_sel):
            return date_ind, skipped

        skipped.append(date_ind)
        date_ind += 1
    
    raise ValueError('Unable to find matching date and offset for "%s", "%s".' % (str(datetime_sel), str(offset_sel)))

def build_dates_and_latlon_coords(start_date, end_date, resolution, bounding_box_name, inclusive_lon=False):
    true_dates, true_offsets = create_true_dates(start_date, end_date, GFS_TIMES, GFS_OFFSETS)

    grid_increment = 1. if resolution == '3' else .5
    lats, lons = GFS_BOUNDING_BOXES[bounding_box_name].make_grid(grid_increment, grid_increment, inclusive_lon)
    lats, lons = lats[:,0], lons[0,:]

    return true_dates, true_offsets, lats, lons

def build_data_arrays(true_dates, lats, lons, variables):
    """ Variables are a tuple (name, dtype). """
    logging.debug('Building data arrays')

    data_arrays = {}
    for name,dtype in variables:
        data_arrays[name] = np.full(shape=(len(true_dates), len(lats), len(lons)), fill_value=np.nan, 
                dtype=dtype)

    return data_arrays

class GfsAggregateYear(luigi.Task):
    raw_data_dir = luigi.parameter.Parameter()
    src_data_dir = luigi.parameter.Parameter()
    dest_data_dir = luigi.parameter.Parameter()

    resolution_sel = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS)
    year_sel = luigi.parameter.YearParameter()

    measurement_sel_name = luigi.parameter.ChoiceParameter(choices=GFS_MEASUREMENT_SEL.keys(), default='default_v1')
    bounding_box_sel_name = luigi.parameter.ChoiceParameter(choices=GFS_BOUNDING_BOXES.keys(), default='global')

    def requires(self):
        self.start_date = dt.date(self.year_sel.year, 1, 1)
        self.end_date = dt.date(self.year_sel.year, 12, 31)

        return GfsGetAvailableFilesList(local_data_dir=self.raw_data_dir, resolution_sel=self.resolution_sel,
                start_date_sel=self.start_date, end_date_sel=self.end_date)

    def run(self):
        logger.debug('Loading available file list')
        with open(self.input().path, 'rb') as fin:
            available_files = pickle.load(fin)

        available_file_names = map(lambda x: os.path.split(x)[1], available_files)
        required_tasks = map(lambda x: create_filter_task_from_file_name(x, self.raw_data_dir, self.src_data_dir, 
            self.measurement_sel_name, self.bounding_box_sel_name, self.year_sel.year), available_file_names)
        
        required_tasks = [task for task in required_tasks if task is not None]

        # Dynamic requirements on filtering all of the available files
        yield required_tasks

        # Create dates and lat/lon for xarray coords
        true_dates, true_offsets, lats, lon = build_dates_and_latlon_coords(self.start_date, self.end_date,
                self.resolution_sel, self.bounding_box_sel_name)

       # Add empty array for each measurement (filled with nans)
        variables = [(s.name, s.dtype) for s in GFS_MEASUREMENT_SEL[self.measurement_sel_name]]
        data_arrays = build_data_arrays(true_dates, lats, lons, variables)

        # For each file (one per task), add the single day slice to the dataset
        units = {}
        date_ind = 0
        for task in required_tasks:
            file_path = task.output().path
            date_sel, time_sel, offset_sel = task.date_sel, task.time_sel, task.offset_sel

            logger.debug(file_path)

            datetime_sel = dt.datetime.combine(date_sel, dt.time(time_sel))
            offset_sel = dt.timedelta(hours=offset_sel)

            date_ind, _ = get_date_ind(true_dates, true_offsets, date_ind, datetime_sel, offset_sel)

            with h5py.File(file_path, 'r') as fin:
                try:
                    for measurement in fin:
                        data_arrays[measurement][date_ind,:,:] = fin[measurement][:]
                        units[measurement] = fin[measurement].attrs['units']
                except ValueError:
                    logger.error('Error aggregating file "%s"' % file_path)
                    continue

            date_ind += 1

        # Output
        logger.debug('Creating dataset')
        with self.output().temporary_path() as temp_output_path:
            with h5py.File(temp_output_path) as ds:
                _ = ds.create_dataset('lat', data=lats)
                _ = ds.create_dataset('lon', data=lons)
                _ = ds.create_dataset('time', data=[str(d).encode('utf8') for d in true_dates])
                _ = ds.create_dataset('offset', data=np.array([o.seconds//3600 for o in true_offsets]))

                for k,v in data_arrays.items():
                    logger.debug('Creating dataset for %s' % k)
                    ds = ds.create_dataset(k, data=v, chunks=True, compression='lzf')
                    ds.attrs['units'] = units.get(k, '')

    def output(self):
        file_name = 'gfsanl_%s_%s_%s_%d.hdf5' % (self.resolution_sel, self.measurement_sel_name, 
                self.bounding_box_sel_name, self.year_sel.year)

        dest_path = os.path.join(self.dest_data_dir, file_name)

        return luigi.LocalTarget(dest_path)

class GfsFilterRegion(luigi.Task):
    data_dir = luigi.parameter.Parameter()

    raw_data_dir = luigi.parameter.Parameter(default='raw/gfs')
    interim_filtered_data_dir = luigi.parameter.Parameter(default='interim/gfs/filtered')
    interim_aggregated_data_dir = luigi.parameter.Parameter(default='interim/gfs/aggregated')
    dest_data_dir = luigi.parameter.Parameter(default='interim/gfs/region')

    measurement_sel_name = luigi.parameter.ChoiceParameter(choices=GFS_MEASUREMENT_SEL.keys(), default='default_v1')
    resolution_sel = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS)

    start_date_sel = luigi.parameter.DateParameter()
    end_date_sel = luigi.parameter.DateParameter()

    bounding_box_sel_name = luigi.parameter.ChoiceParameter(choices=GFS_BOUNDING_BOXES.keys())

    def requires(self):
        start_year = self.start_date_sel.year
        end_year = self.end_date_sel.year
        years = range(start_year, end_year+1)

        return [GfsAggregateYear(raw_data_dir=os.path.join(self.data_dir, self.raw_data_dir, self.resolution_sel),
            src_data_dir=os.path.join(self.data_dir, self.interim_filtered_data_dir, self.resolution_sel),
            dest_data_dir=os.path.join(self.data_dir, self.interim_aggregated_data_dir, self.resolution_sel),
            resolution_sel=self.resolution_sel, 
            measurement_sel_name=self.measurement_sel_name, year_sel=dt.date(year, 1, 1)) for year in years]

    def run(self):
        bounding_box = GFS_BOUNDING_BOXES[self.bounding_box_sel_name]
        lat_min, lat_max, lon_min, lon_max = bounding_box.get()

        # Create xarray dataset
        true_dates, true_offsets, lats, lons = build_dates_and_latlon_coords(self.start_date_sel, self.end_date_sel,
                self.resolution_sel, self.bounding_box_sel_name, inclusive_lon=True)

        variables = [(s.name, s.dtype) for s in GFS_MEASUREMENT_SEL[self.measurement_sel_name]]
        data_arrays = build_data_arrays(true_dates, lats, lons, variables)
        data_arrays = {k: (['time', 'y', 'x'], v) for k,v in data_arrays.items()}

        ds = xr.Dataset(data_arrays, coords={'lat': (['y'], lats), 'lon': (['x'], lons),
            'time': true_dates, 'offset': (['time'], true_offsets)})

        # Iterate over each aggregated year
        for path in [f.path for f in self.input()]:
            logger.debug('Adding "%s"' % path)
            with h5py.File(path) as ds_in:
                time_in= pd.to_datetime([t.decode('utf-8') for t in ds_in['time'][:]])
                offset_in = [dt.timedelta(hours=int(o)) for o in ds_in['offset']]
                lats_in, lons_in = ds_in['lat'][:], ds_in['lon'][:]

                year = time_in[0].year
                cur_start_date, cur_end_date = dt.date(year, 1, 1), dt.date(year, 12, 31)

                cur_start_date = cur_start_date if cur_start_date >= self.start_date_sel else self.start_date_sel 
                cur_end_date = cur_end_date if cur_end_date <= self.end_date_sel else self.end_date_sel 

                cur_start_date, cur_end_date = pd.to_datetime([cur_start_date, cur_end_date + dt.timedelta(days=1)])

                # Select lat/lon and date range from input
                time_ind = (time_in >= cur_start_date) & (time_in < cur_end_date)
                lat_ind = (lats_in >= lat_min) & (lats_in <= lat_max)
                lon_ind = (lons_in >= lon_min) & (lons_in <= lon_max)

                # Select time ind for the dataset
                cur_start_date, cur_end_date = np.datetime64(cur_start_date), np.datetime64(cur_end_date)
                time_ind_ds = (ds.time >= cur_start_date) & (ds.time < cur_end_date)

                # Filter dataset
                for measurement in ds.data_vars.keys():
                    """
                    h5py does not support multiple advanced slices at once and lat/lon slicing will generally
                    reduce the size to be loaded faster than date slices (usually we will take all dates for a year)
                    """
                    val_in = ds_in[measurement][:, lat_ind, :]
                    val_in = val_in[:, :, lon_ind]
                    val_in = val_in[time_ind, :, :]

                    ds[measurement][time_ind_ds, :, :] = val_in
                    ds[measurement].attrs['units'] = ds_in[measurement].attrs.get('units', '')

        # Save selected ranges
        with self.output().temporary_path() as temp_output_path:
            encoding = {k: {'zlib': True, 'complevel': 1} for k in ds.data_vars.keys()}
            ds.to_netcdf(temp_output_path, engine='netcdf4', encoding=encoding)

    def output(self):
        date_fmt = '%m%d%Y'

        start_date_str = self.start_date_sel.strftime(date_fmt)
        end_date_str =  self.end_date_sel.strftime(date_fmt)

        fn = 'gfs_%s_%s_%s_%s.nc' % (self.resolution_sel, self.bounding_box_sel_name, start_date_str, end_date_str)
        dest_path = os.path.join(self.data_dir, self.dest_data_dir, self.resolution_sel, fn)

        return luigi.LocalTarget(dest_path)
