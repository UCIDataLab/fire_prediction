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

import parse # For latlon param type
from helper.geometry import LatLonBoundingBox # For latlon param type
from helper.date_util import daterange_days

from .pipeline_params import (REGION_BOUNDING_BOXES, GFS_MEASUREMENT_SEL, GFS_TIMES, GFS_OFFSETS, GFS_RESOLUTIONS, 
        GFS_RAW_DATA_DIR, GFS_FILTERED_DATA_DIR, GFS_AGGREGATED_DATA_DIR, GFS_REGION_DATA_DIR,
        GFS_SERVER_NAME, GFS_SERVER_USERNAME, GFS_SERVER_PASSWORD, GFS_SERVER_DATA_DIR, ERA_TIMES, ERA_OFFSETS, 
        ERA_MEASUREMENT_SEL)
from .pipeline_helper import (logger, check_spanning_file, check_date_str_spanning, change_data_dir_path, FtpFile, 
        build_dates_and_latlon_coords, build_data_arrays)

year_month_dir_fmt = "%d%.2d"
year_month_day_dir_fmt = "%d%.2d%.2d"
grib_file_fmt_half_deg = "gfsanl_4_%s_%.2d00_%.3d.grb2"
grib_file_fmt_one_deg = "gfsanl_3_%s_%.2d00_%.3d.grb"

def_name_conversion_dict = {'Surface air relative humidity': 'humidity', '2 metre relative humidity': 'humidity',
        'Relative humidity': 'humidity', '10 metre U wind component': 'U wind component', 
        '10 metre V wind component': 'V wind component', 'Convective available potential energy': 'cape', 
        'Planetary boundary layer height': 'pbl_height', 'Volumetric soil moisture content': 'soil_moisture' }

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

class GfsFtpFileDownload(luigi.Task):
    data_dir = luigi.parameter.Parameter()
    dest_data_dir = luigi.parameter.Parameter(default=GFS_RAW_DATA_DIR)

    server_name = luigi.parameter.Parameter(default=GFS_SERVER_NAME)
    server_username = luigi.parameter.Parameter(default=GFS_SERVER_USERNAME, significant=False)
    server_password = luigi.parameter.Parameter(default=GFS_SERVER_PASSWORD, significant=False)
    server_data_dir = luigi.parameter.Parameter(default=GFS_SERVER_DATA_DIR)

    resolution = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS)
    date = luigi.parameter.DateParameter()
    time = luigi.parameter.ChoiceParameter(choices=GFS_TIMES, var_type=int)
    offset = luigi.parameter.ChoiceParameter(choices=GFS_OFFSETS, var_type=int)

    resources = {'ftp': 1}

    def requires(self):
        file_path = build_gfs_server_file_path(self.server_data_dir, self.resolution, self.date, 
                self.time, self.offset)
        return FtpFile(server_name=self.server_name, server_username=self.server_username,
                server_password=self.server_password, file_path=file_path)

    def run(self):
        # Copy ftp file from server to local dest
        with self.output().temporary_path() as temp_output_path:
            self.input().get(temp_output_path)

    def output(self):
        dest_dir = os.path.join(self.data_dir, self.dest_data_dir, self.resolution)
        dest_path = change_data_dir_path(self.server_data_dir, dest_dir, self.input().path)

        return luigi.LocalTarget(dest_path)

class GfsFilterMeasurements(luigi.Task):
    data_dir = luigi.parameter.Parameter()
    src_data_dir = luigi.parameter.Parameter(default=GFS_RAW_DATA_DIR)
    dest_data_dir = luigi.parameter.Parameter(default=GFS_FILTERED_DATA_DIR)

    resolution = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS)
    date = luigi.parameter.DateParameter()
    time = luigi.parameter.ChoiceParameter(choices=GFS_TIMES, var_type=int)
    offset = luigi.parameter.ChoiceParameter(choices=GFS_OFFSETS, var_type=int)

    measurement_sel_name = luigi.parameter.ChoiceParameter(choices=GFS_MEASUREMENT_SEL.keys())
    bounding_box_name = luigi.parameter.ChoiceParameter(choices=REGION_BOUNDING_BOXES.keys())

    def requires(self):
        return GfsFtpFileDownload(data_dir=self.data_dir, resolution=self.resolution, date=self.date, time=self.time,
                offset=self.offset)

    def run(self):
        # Read grib file and extract selected measurements
        with pygrib.open(self.input().path) as fin:
            selections = GFS_MEASUREMENT_SEL[self.measurement_sel_name]
            bounding_box = REGION_BOUNDING_BOXES[self.bounding_box_name]

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
        dest_path = change_data_dir_path(os.path.join(self.data_dir, self.src_data_dir, self.resolution), 
                os.path.join(self.data_dir, self.dest_data_dir, self.resolution), self.input().path)
        dest_path, _ = os.path.splitext(dest_path)
        dest_path += '_%s_%s' % (self.measurement_sel_name, self.bounding_box_name)
        dest_path += '.hdf5'
        return luigi.LocalTarget(dest_path)

class GfsGetAvailableFilesList(luigi.Task):
    data_dir = luigi.parameter.Parameter()
    dest_data_dir = luigi.parameter.Parameter(default=GFS_RAW_DATA_DIR)

    server_name = luigi.parameter.Parameter(default=GFS_SERVER_NAME)
    server_username = luigi.parameter.Parameter(default=GFS_SERVER_USERNAME, significant=False)
    server_password = luigi.parameter.Parameter(default=GFS_SERVER_PASSWORD, significant=False)
    server_data_dir = luigi.parameter.Parameter(default=GFS_SERVER_DATA_DIR)

    resolution = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS)
    start_date = luigi.parameter.DateParameter()
    end_date = luigi.parameter.DateParameter()

    resources = {'ftp': 1}

    def run(self):
        ftp = FTP(self.server_name)
        ftp.login(self.server_username, self.server_password)

        available_files = []

        for date in daterange_days(self.start_date, self.end_date + dt.timedelta(1)):
            day_dir_path = build_gfs_server_file_path(self.server_data_dir, self.resolution, date)
            try:
                files = ftp.nlst(day_dir_path)
            except:
                logger.debug('Misssing Day: resolution %s year %d month %d day %d' % (self.resolution, date.year, date.month, date.day))
                continue

            for time,offset in [(t,o) for t in GFS_TIMES for o in GFS_OFFSETS]:
                file_path = build_gfs_server_file_path(self.server_data_dir, self.resolution, date, time, offset)
                dirname, file_name = os.path.split(file_path)

                if file_path in files or file_name in files: 
                    available_files.append(file_path)
                else:
                    logger.debug('Missing File: resolution %s year %d month %d day %d time %d offset %d not on server'
                            % (self.resolution, date.year, date.month, date.day, time, offset))

        # Clean-up
        ftp.quit()

        with self.output().temporary_path() as temp_output_path:
            with open(temp_output_path, 'wb') as fout:
                pickle.dump(available_files, fout, protocol=pickle.HIGHEST_PROTOCOL)

    def output(self):
        date_fmt = '%m%d%Y'

        start_date_str = self.start_date.strftime(date_fmt)
        end_date_str =  self.end_date.strftime(date_fmt)

        file_name = 'gfsanl_%s_available_%s-%s.pkl' % (self.resolution, start_date_str, end_date_str)
        dest_path = os.path.join(self.data_dir, self.dest_data_dir, self.resolution, file_name)

        return luigi.LocalTarget(dest_path)

def create_filter_task_from_file_name(fn, data_dir, measurement_sel_name, bounding_box_name):
    fmt = 'gfsanl_{resolution}_{date}_{time:d}_{offset:d}.{}' 
    p = parse.parse(fmt, fn)

    year,month,day = int(p['date'][:4]), int(p['date'][4:6]), int(p['date'][6:8])

    resolution = p['resolution']
    date = dt.date(year, month, day)
    time=p['time']//100
    offset=p['offset']

    task = GfsFilterMeasurements(data_dir=data_dir, resolution=resolution, date=date, time=time, offset=offset,
            measurement_sel_name=measurement_sel_name, bounding_box_name=bounding_box_name)
    
    return task

def get_date_ind(true_dates, true_offsets, date_ind, datetime, offset):
    """ Iterate throught list of dates/offsets to find first match of given date/offset starting at date_ind. """

    skipped = []
    while True:
        if (true_dates[date_ind] == datetime) and (true_offsets[date_ind]==offset):
            return date_ind, skipped

        skipped.append(date_ind)
        date_ind += 1
    
    raise ValueError('Unable to find matching date and offset for "%s", "%s".' % (str(datetime), str(offset)))
class GfsAggregateYear(luigi.Task):
    resources = {'memory': 60}

    data_dir = luigi.parameter.Parameter()
    dest_data_dir = luigi.parameter.Parameter(default=GFS_AGGREGATED_DATA_DIR)

    resolution = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS)

    start_date = luigi.parameter.DateParameter()
    end_date = luigi.parameter.DateParameter()

    measurement_sel_name = luigi.parameter.ChoiceParameter(choices=GFS_MEASUREMENT_SEL.keys(), default='default_v1')
    bounding_box_name = luigi.parameter.ChoiceParameter(choices=REGION_BOUNDING_BOXES.keys(), default='global')

    def requires(self):
        assert(self.start_date.year == self.end_date.year)

        self.start_date = self.start_date if self.start_date else dt.date(self.start_date.year, 1, 1) 
        self.end_date = self.end_date if self.end_date else dt.date(self.start_date.year, 12, 31)

        return GfsGetAvailableFilesList(data_dir=self.data_dir, resolution=self.resolution,
                start_date=self.start_date, end_date=self.end_date)

    def run(self):
        logger.debug('Loading available file list')
        with open(self.input().path, 'rb') as fin:
            available_files = pickle.load(fin)

        available_file_names = map(lambda x: os.path.split(x)[1], available_files)
        required_tasks = map(lambda x: create_filter_task_from_file_name(x, self.data_dir, self.measurement_sel_name, 
            self.bounding_box_name), available_file_names)
        
        required_tasks = [task for task in required_tasks if task is not None]

        # Dynamic requirements on filtering all of the available files
        yield required_tasks

        # Create dates and lat/lon for xarray coords
        bounding_box = REGION_BOUNDING_BOXES[self.bounding_box_name]
        true_dates, true_offsets, lats, lons = build_dates_and_latlon_coords(self.start_date, self.end_date,
                self.resolution, bounding_box, GFS_TIMES, GFS_OFFSETS)

       # Add empty array for each measurement (filled with nans)
        variables = [(s.name, s.dtype) for s in GFS_MEASUREMENT_SEL[self.measurement_sel_name]]
        data_arrays = build_data_arrays(true_dates, lats, lons, variables)

        # For each file (one per task), add the single day slice to the dataset
        units = {}
        date_ind = 0
        for task in required_tasks:
            file_path = task.output().path
            date, time, offset = task.date, task.time, task.offset

            logger.debug(file_path)

            datetime = dt.datetime.combine(date, dt.time(time))
            offset = dt.timedelta(hours=offset)

            date_ind, _ = get_date_ind(true_dates, true_offsets, date_ind, datetime, offset)

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
            with h5py.File(temp_output_path) as f:
                _ = f.create_dataset('lat', data=lats)
                _ = f.create_dataset('lon', data=lons)
                _ = f.create_dataset('time', data=[str(d).encode('utf8') for d in true_dates])
                _ = f.create_dataset('offset', data=np.array([o.seconds//3600 for o in true_offsets]))

                for k,v in data_arrays.items():
                    logger.debug('Creating dataset for %s' % k)
                    ds = f.create_dataset(k, data=v, chunks=True, compression='lzf')
                    ds.attrs['units'] = units.get(k, '')

    def output(self):
        date_fmt = '%m%d'
        start_date_str = self.start_date.strftime(date_fmt)
        end_date_str =  self.end_date.strftime(date_fmt)

        file_name = 'gfsanl_%s_%s_%s_%d_%s_%s.hdf5' % (self.resolution, self.measurement_sel_name, 
                self.bounding_box_name, self.start_date.year, start_date_str, end_date_str)

        dest_path = os.path.join(self.data_dir, self.dest_data_dir, self.resolution, file_name)

        return luigi.LocalTarget(dest_path)


def filter_region_span_check_func(test_fn, cur_fn):
    fmt = 'gfs_{resolution}_{bb_name}_{start_date}_{end_date}.nc' 
    test_p = parse.parse(fmt, test_fn)
    cur_p = parse.parse(fmt, cur_fn)

    check = True
    check = check and (test_p['resolution']==cur_p['resolution'])
    check = check and (test_p['bb_name']==cur_p['bb_name'])

    # Exit early if files don't match
    if not check:
        return check

    # Check dates
    date_fmt = '%Y%m%d'
    check = check and check_date_str_spanning(test_p['start_date'], cur_p['start_date'], date_fmt)
    check = check and check_date_str_spanning(test_p['end_date'], cur_p['end_date'], date_fmt)

    return check

class EraAggregateYear(luigi.ExternalTask):
    resources = {'memory': 60}

    data_dir = luigi.parameter.Parameter()
    dest_data_dir = luigi.parameter.Parameter(default=GFS_AGGREGATED_DATA_DIR)

    resolution = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS)

    start_date = luigi.parameter.DateParameter()
    end_date = luigi.parameter.DateParameter()

    measurement_sel_name = luigi.parameter.ChoiceParameter(choices=GFS_MEASUREMENT_SEL.keys(), default='default_v1')
    bounding_box_name = luigi.parameter.ChoiceParameter(choices=REGION_BOUNDING_BOXES.keys(), default='global')

    def output(self):
        dest_path = os.path.join(self.data_dir, self.dest_data_dir, 'era',
                'eraanl_default_v1_alaska_%d_0101_1231.hdf5' % self.start_date.year)
        return luigi.LocalTarget(dest_path)

class GfsFilterRegion(luigi.Task):
    data_dir = luigi.parameter.Parameter()
    dest_data_dir = luigi.parameter.Parameter(default=GFS_REGION_DATA_DIR)

    measurement_sel_name = luigi.parameter.ChoiceParameter(choices=GFS_MEASUREMENT_SEL.keys(), default='default_v1')
    resolution = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS)

    start_date = luigi.parameter.DateParameter()
    end_date = luigi.parameter.DateParameter()

    bounding_box_name = luigi.parameter.ChoiceParameter(choices=REGION_BOUNDING_BOXES.keys())

    use_era = luigi.parameter.BoolParameter()

    def requires(self):
        start_year, end_year = self.start_date.year, self.end_date.year
        years = range(start_year, end_year+1)
        
        tasks = []
        for year in years:
            start_date = dt.date(year, 1, 1)
            end_date = dt.date(year, 12, 31) 

            if year==start_year:
                start_date = dt.date(year, self.start_date.month, self.start_date.day)

            if year==end_year:
                end_date = dt.date(year, self.end_date.month, self.end_date.day) 

            if self.use_era:
                tasks.append(EraAggregateYear(data_dir=self.data_dir, resolution=self.resolution, 
                    measurement_sel_name=self.measurement_sel_name, start_date=start_date, 
                    end_date=end_date))
            else:
                tasks.append(GfsAggregateYear(data_dir=self.data_dir, resolution=self.resolution, 
                    measurement_sel_name=self.measurement_sel_name, start_date=start_date, 
                    end_date=end_date))

        return tasks

    def run(self):
        bounding_box = REGION_BOUNDING_BOXES[self.bounding_box_name]
        lat_min, lat_max, lon_min, lon_max = bounding_box.get()

        # Create xarray dataset
        bounding_box = REGION_BOUNDING_BOXES[self.bounding_box_name]
        if self.use_era:
            true_dates, true_offsets, lats, lons = build_dates_and_latlon_coords(self.start_date, self.end_date,
                    self.resolution, bounding_box, ERA_TIMES, ERA_OFFSETS, inclusive_lon=True)
            variables = [(s.name, s.dtype) for s in ERA_MEASUREMENT_SEL[self.measurement_sel_name]]
        else:
            true_dates, true_offsets, lats, lons = build_dates_and_latlon_coords(self.start_date, self.end_date,
                    self.resolution, bounding_box, GFS_TIMES, GFS_OFFSETS, inclusive_lon=True)
            variables = [(s.name, s.dtype) for s in GFS_MEASUREMENT_SEL[self.measurement_sel_name]]

        data_arrays = build_data_arrays(true_dates, lats, lons, variables)
        data_arrays = {k: (['time', 'y', 'x'], v) for k,v in data_arrays.items()}

        ds = xr.Dataset(data_arrays, coords={'lat': (['y'], lats), 'lon': (['x'], lons),
            'time': true_dates, 'offset': (['time'], true_offsets)})

        # Iterate over each aggregated year
        for path in [f.path for f in self.input()]:
            logger.debug('Adding "%s"' % path)
            if self.use_era:
                ds_in = xr.open_dataset(path)
                if self.use_era:
                    time_in= pd.to_datetime(np.array(ds_in['time']))
                else:
                    time_in= pd.to_datetime([t.decode('utf-8') for t in np.array(ds_in['time'])])
                offset_in = [dt.timedelta(hours=int(o)) for o in ds_in['offset']]
                lats_in, lons_in = np.array(ds_in['lat']), np.array(ds_in['lon'])

                year = time_in[0].year
                cur_start_date, cur_end_date = dt.date(year, 1, 1), dt.date(year, 12, 31)

                cur_start_date = cur_start_date if cur_start_date >= self.start_date else self.start_date 
                cur_end_date = cur_end_date if cur_end_date <= self.end_date else self.end_date 

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


            else:
                with h5py.File(path) as ds_in:
                    time_in= pd.to_datetime([t.decode('utf-8') for t in ds_in['time'][:]])
                    offset_in = [dt.timedelta(hours=int(o)) for o in ds_in['offset']]
                    lats_in, lons_in = ds_in['lat'][:], ds_in['lon'][:]

                    year = time_in[0].year
                    cur_start_date, cur_end_date = dt.date(year, 1, 1), dt.date(year, 12, 31)

                    cur_start_date = cur_start_date if cur_start_date >= self.start_date else self.start_date 
                    cur_end_date = cur_end_date if cur_end_date <= self.end_date else self.end_date 

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
        date_fmt = '%Y%m%d'

        start_date_str = self.start_date.strftime(date_fmt)
        end_date_str =  self.end_date.strftime(date_fmt)

        if self.use_era:
            fn = 'era_%s_%s_%s_%s.nc' % (self.resolution, self.bounding_box_name, start_date_str, end_date_str)
        else:
            fn = 'gfs_%s_%s_%s_%s.nc' % (self.resolution, self.bounding_box_name, start_date_str, end_date_str)

        dest_path = os.path.join(self.data_dir, self.dest_data_dir, self.resolution, fn)

        dest_path = check_spanning_file(dest_path, filter_region_span_check_func)

        return luigi.LocalTarget(dest_path)


