import numpy as np
import luigi
import os
import xarray as xr
import pandas as pd
import logging

from gfs_pipeline import GfsFilterRegion, GFS_RESOLUTIONS
from data.gfs_choices import GFS_BOUNDING_BOXES

WEATHER_FILL_METH = ['integrate', 'mean', 'interpolate', 'drop']

logger = logging.getLogger('pipeline')

class WeatherFillMissingValues(luigi.Task):
    """ 
    Fill missing weather values. 
    
    Options include integrating multiple sources (e.g. .5 and 1.), interpolation, and  using mean values.
    """
    data_dir = luigi.parameter.Parameter()
    dest_data_dir = luigi.parameter.Parameter(default='interim/gfs/filled')

    start_date = luigi.parameter.DateParameter()
    end_date = luigi.parameter.DateParameter()

    resolution = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS)
    bounding_box_name = luigi.parameter.ChoiceParameter(choices=GFS_BOUNDING_BOXES.keys())

    fill_method = luigi.parameter.ChoiceParameter(choices=WEATHER_FILL_METH)

    def requires(self):
        tasks = {res: GfsFilterRegion(data_dir=self.data_dir, resolution_sel=res, start_date_sel=self.start_date,
                end_date_sel=self.end_date, bounding_box_sel_name=self.bounding_box_name) for res in GFS_RESOLUTIONS}

        if self.fill_method == 'integrate':
            return tasks
        else:
            return {self.resolution: tasks[self.resolution]}

    def run(self):
        data_in = {k: xr.open_dataset(v.path) for k,v in self.input().items()}

        data_filled = self.fill_data(data_in)

        with self.output().temporary_path() as temp_output_path:
            data_filled.to_netcdf(temp_output_path, engine='netcdf4')

    def output(self):
        fn_in, _ = os.path.splitext(os.path.split(self.input()[self.resolution].path)[1])
        fn = fn_in + '_%s' % self.fill_method + '.nc'

        dest_path = os.path.join(self.data_dir, self.dest_data_dir, fn)

        return luigi.LocalTarget(dest_path)

    def fill_data(self, data_in):
        data_filled = data_in[self.resolution]

        if self.fill_method == 'integrate':
            other_resolution = '3' if self.resolution == '4' else '4'
            data_filler = data_in[other_resolution]

            for measurement in data_filled.data_vars.keys():
                nan_days = np.any(data_filled[measurement].isnull(), axis=(1,2))
                non_nan_days_filler = np.all(data_filler[measurement].notnull(), axis=(1,2))

                fill_days = nan_days & non_nan_days_filler

                filler = data_filler[measurement][fill_days]
                filler = upsample_spatial(data_filled[measurement].shape[1:], filler)

                data_filled[measurement][fill_days] = filler

                # Fill any remaining nans with the cell mean (TODO: Support alternatives)
                unfilled_days = nan_days & (~non_nan_days_filler)
                logger.debug('%s -- filled by mean -- %d' % (measurement, np.sum(unfilled_days)))

                measurement_mean = np.nanmean(data_filled[measurement], axis=0)
                data_filled[measurement][unfilled_days] = measurement_mean

                if np.any(np.isnan(data_filled[measurement])):
                    logger.warn('%s -- NaNs still present' % measurement)

        else:
            raise NotImplementedError()

        return data_filled

def upsample_spatial(target_shape, data):
    target_shape = (data.shape[0],) + target_shape
    upsampled = np.empty(target_shape, dtype=data.dtype)

    # TODO: Look into this upsampling more
    x_ratio = np.ceil(target_shape[1] / data.shape[1]).astype(np.int32)
    y_ratio = np.ceil(target_shape[2] / data.shape[2]).astype(np.int32)

    for (x,y) in [(x,y) for x in range(target_shape[1]) for y in range(target_shape[2])]:
        upsampled[:,x,y] = data[:, x//x_ratio, y//y_ratio]

    return upsampled

class WeatherGridGeneration(luigi.Task):
    """
    Final preparation of weather grid so it can be used to generate datasets.

    Includes computing derived variables (e.g. wind speed, 24-hour rain accumulation) and discarding
    unnecessary information (e.g. the 3 and 6 hour offsets from GFS).
    """
    data_dir = luigi.parameter.Parameter()
    dest_data_dir = luigi.parameter.Parameter(default='interim/gfs/grid')

    start_date = luigi.parameter.DateParameter()
    end_date = luigi.parameter.DateParameter()

    resolution = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS)
    bounding_box_name = luigi.parameter.ChoiceParameter(choices=GFS_BOUNDING_BOXES.keys())

    fill_method = luigi.parameter.ChoiceParameter(choices=WEATHER_FILL_METH)

    def requires(self):
        return WeatherFillMissingValues(data_dir=self.data_dir, start_date=self.start_date, end_date=self.end_date,
                resolution=self.resolution, bounding_box_name=self.bounding_box_name, fill_method=self.fill_method)

    def run(self):
        data = xr.open_dataset(self.input().path)

        logger.debug('Integrating rain')
        data = self.integrate_rain(data)
        logger.debug('Discarding')
        data = self.discard_offset_measurments(data)
        logger.debug('Computing wind')
        data = self.compute_wind_speed(data)

        logger.debug('Saving data')
        with self.output().temporary_path() as temp_output_path:
            data.to_netcdf(temp_output_path, engine='netcdf4')

    def output(self):
        fn_in, _ = os.path.splitext(os.path.split(self.input().path)[1])
        fn = fn_in + '_grid' + '.nc'

        dest_path = os.path.join(self.data_dir, self.dest_data_dir, fn)

        return luigi.LocalTarget(dest_path)

    def integrate_rain(self, data):
        rain_da = data['precipitation']

        integrated_rain = np.empty(rain_da.shape)
        for i in range(rain_da.shape[0], 3):
            if i < 12:
                integrated_rain[i] = 0 # TODO: Use available info (even if its incomplete)
            else:
                if i % 1000:
                    logger.debug('Current rain ind:', str(i))
                integrated_rain[i] = rain_da[i-10] + rain_da[i-7] + rain_da[i-4] + rain_da[i-1]

        rain_da[:] = integrated_rain

        data = data.rename({'precipitation': 'precipitation_24hr'})

        return data

    def discard_offset_measurments(self, data):
        #offset = pd.to_timedelta(data.offset)
        #non_offset_inds = offset.seconds == 0

        # Every third time step is non-offset measurement
        non_offset_slice = slice(0, None, 3)

        logger.debug('Building data arrays')
        data_arrays = {}
        encodings = {}
        for measurement in data.data_vars.keys():
            data_arrays[measurement] = (['time', 'y', 'x'], np.array(data[measurement])[non_offset_slice],
                    data[measurement].attrs)
            encodings[measurement] = data[measurement].encoding

        logger.debug('Building coords')
        time = np.array(data.time)[non_offset_slice]
        lat = np.array(data.lat)
        lon = np.array(data.lon)

        logger.debug('Building dataset')
        ds_new = xr.Dataset(data_arrays, coords={'lat': (['y'], lat), 'lon': (['x'], lon),
            'time': time}, attrs=data.attrs)

        for measurement in ds_new.data_vars.keys():
            ds_new[measurement].encoding = encodings[measurement]

        return ds_new

    def compute_wind_speed(self, data):
        u_wind = data['u_wind_component']
        v_wind = data['v_wind_component']

        logger.debug('Loading wind data')
        u_wind_speed = np.array(u_wind)
        v_wind_speed = np.array(v_wind)

        np.seterr(invalid='raise')
        logger.debug('Computing wind magnitude')
        print(np.any(u_wind_speed==np.nan), np.any(v_wind_speed==np.nan))
        squared_vals = np.square(u_wind_speed) + np.square(v_wind_speed)
        print(squared_vals.dtype, np.any(squared_vals<0), np.any(squared_vals==np.nan))
        for v in squared_vals:
            try:
                wind_speed = np.sqrt(v)
            except:
                print('Invalid', str(v))

        logger.debug('Updating wind values')
        u_wind[:] = wind_speed

        logger.debug('Dropping wind values')
        data = data.drop(['v_wind_component'])

        logger.debug('Renaming wind values')
        data = data.rename({'u_wind_component': 'wind_speed'})

        return data

