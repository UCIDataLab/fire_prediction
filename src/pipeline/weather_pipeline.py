import logging
import os
import time

import luigi
import numpy as np
import xarray as xr

from src.helper.geometry import upsample_spatial
from .gfs_pipeline import GfsFilterRegion
from .pipeline_params import REGION_BOUNDING_BOXES, GFS_RESOLUTIONS, WEATHER_FILL_METHOD, GFS_OFFSETS

# 48 timesteps (aprox. 4 days)
INTERP_FILL_LIM = None
NUM_TIMES_PER_DAY = 12

logger = logging.getLogger('pipeline')


def repeat_spatially(arr, shape):
    arr_new = arr[:, None, None]
    arr_new = np.repeat(arr_new, shape[0], axis=1)
    arr_new = np.repeat(arr_new, shape[1], axis=2)

    return arr_new


def grid_interpolate_nan_days(data, nan_inds):
    """ Interpolate regular grid data at sample points nan_inds. """
    nan_days = np.arange(data.shape[0])[nan_inds]

    days = np.arange(data.shape[0])[~nan_inds]
    lats = np.arange(data.shape[1])
    lons = np.arange(data.shape[2])

    # points = (days, lats, lons)
    # sample = np.array(list(itertools.product(days, lats, lons)))
    # interp_values = interp.interpn(points, data[~nan_inds], sample)

    interp_values = np.empty((np.sum(nan_inds), data.shape[1], data.shape[2]), dtype=data.dtype)
    interp_values.fill(np.nan)
    for y in lats:
        for x in lons:
            interp_values[:, y, x] = np.interp(nan_days, days, data[~nan_inds, y, x])

    # return np.reshape(interp_values, (len(sample), data.shape[1], data.shape[2]))
    return interp_values


class WeatherFillMissingValues(luigi.Task):
    """ 
    Fill missing weather values. 
    
    Options include integrating multiple sources (e.g. .5 and 1.), interpolation, and  using mean values.
    """
    data_dir: str = luigi.parameter.Parameter()
    dest_data_dir: str = luigi.parameter.Parameter(default='interim/gfs/filled')

    start_date = luigi.parameter.DateParameter()
    end_date = luigi.parameter.DateParameter()

    resolution: str = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS, var_type=str)
    bounding_box_name = luigi.parameter.ChoiceParameter(choices=REGION_BOUNDING_BOXES.keys(), var_type=str)

    fill_method = luigi.parameter.ChoiceParameter(choices=WEATHER_FILL_METHOD)

    use_era = luigi.parameter.BoolParameter()

    def requires(self):
        if self.use_era:
            tasks = {'4': GfsFilterRegion(data_dir=self.data_dir, resolution='4', start_date=self.start_date,
                                          end_date=self.end_date, bounding_box_name=self.bounding_box_name,
                                          use_era=self.use_era)}

            return {'4': tasks[self.resolution]}

        else:
            tasks = {res: GfsFilterRegion(data_dir=self.data_dir, resolution=res, start_date=self.start_date,
                                          end_date=self.end_date, bounding_box_name=self.bounding_box_name) for res in
                     GFS_RESOLUTIONS}

            if (self.fill_method == 'integrate_mean') or (self.fill_method == 'integrate_interp'):
                return tasks
            else:
                return {self.resolution: tasks[self.resolution]}

    def run(self):
        data_in = {k: xr.open_dataset(v.path) for k, v in self.input().items()}

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

        in_filled = np.zeros(data_filled['temperature'].values.shape[0], dtype=bool)
        interpolated = np.zeros(data_filled['temperature'].values.shape[0], dtype=bool)
        mean_filled = np.zeros(data_filled['temperature'].values.shape[0], dtype=bool)

        # TODO: Support not interpolating certain measurements (e.g. land sea mask)
        for measurement in data_filled.data_vars.keys():
            if self.fill_method == 'drop':
                raise NotImplementedError()

            if (self.fill_method == 'integrate_mean') or (self.fill_method == 'integrate_interp'):
                other_resolution = '3' if self.resolution == '4' else '4'
                data_filler = data_in[other_resolution]

                nan_days = np.any(data_filled[measurement].isnull(), axis=(1, 2))
                non_nan_days_filler = np.all(data_filler[measurement].notnull(), axis=(1, 2))

                fill_days = nan_days & non_nan_days_filler

                if measurement in ['temperature', 'humidity', 'u_wind_component', 'v_wind_component']:
                    print(fill_days[:10])
                    print('b', np.sum(in_filled))
                    in_filled |= fill_days
                    print('a', np.sum(in_filled))

                filler = data_filler[measurement][fill_days]
                filler = upsample_spatial(data_filled[measurement].shape[1:], filler)

                data_filled[measurement][fill_days] = filler

            if (self.fill_method == 'interpolate') or (self.fill_method == 'integrate_interp'):
                # Fill remaining nans with linear interp
                unfilled_days = np.any(data_filled[measurement].isnull(), axis=(1, 2))
                logger.debug('{} -- to fill by lin. interpolation -- {:d}'.format(measurement, np.sum(unfilled_days)))

                if measurement in ['temperature', 'humidity', 'u_wind_component', 'v_wind_component']:
                    print(unfilled_days[:10])
                    print('b2', np.sum(interpolated))
                    interpolated |= unfilled_days
                    print('a2', np.sum(interpolated))

                # Interpolate each time of day (and offset) separately
                for i in range(NUM_TIMES_PER_DAY):
                    # data_filled[measurement] = data_filled[measurement].interpolate_na(dim='time',
                    #        use_coordinate=False, limit=INTERP_FILL_LIM)

                    nan_days = np.any(np.isnan(data_filled[measurement][i::12]), axis=(1, 2))

                    # If no points or all points are none, do not perform interp
                    if (not np.all(nan_days)) and (not np.all(~nan_days)):
                        data_filled[measurement][i::12][nan_days] = grid_interpolate_nan_days(
                            data_filled[measurement][i::12], nan_days)

            # Fill any remaining nans with the cell mean 
            if np.any(np.isnan(data_filled[measurement])):
                unfilled_days = np.any(data_filled[measurement].isnull(), axis=(1, 2))
                logger.debug('{} -- to fill by mean -- {:d}'.format(measurement, np.sum(unfilled_days)))

                if measurement in ['temperature', 'humidity', 'u_wind_component', 'v_wind_component']:
                    mean_filled |= unfilled_days

                measurement_mean = np.nanmean(data_filled[measurement], axis=0)
                data_filled[measurement][unfilled_days] = measurement_mean

            if np.any(np.isnan(data_filled[measurement])):
                logger.warning('%s -- NaNs still present' % measurement)

            spatial_shape = data_filled['temperature'].values.shape[1:]
            data_filled.update({'in_filled': (('time', 'y', 'x'), repeat_spatially(in_filled, spatial_shape))})
            data_filled.update({'interpolated': (('time', 'y', 'x'), repeat_spatially(interpolated, spatial_shape))})
            data_filled.update({'mean_filled': (('time', 'y', 'x'), repeat_spatially(mean_filled, spatial_shape))})

        return data_filled


class WeatherGridGeneration(luigi.Task):
    """
    Final preparation of weather grid so it can be used to generate datasets.

    Includes computing derived variables (e.g. wind speed, 24-hour rain accumulation) and discarding
    unnecessary information (e.g. the 3 and 6 hour offsets from GFS).
    """
    data_dir: str = luigi.parameter.Parameter()
    dest_data_dir: str = luigi.parameter.Parameter(default='interim/gfs/grid')

    start_date = luigi.parameter.DateParameter()
    end_date = luigi.parameter.DateParameter()

    resolution = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS)
    bounding_box_name = luigi.parameter.ChoiceParameter(choices=REGION_BOUNDING_BOXES.keys())

    fill_method = luigi.parameter.ChoiceParameter(choices=WEATHER_FILL_METHOD)

    rain_offset: int = luigi.parameter.NumericalParameter(var_type=int, min_value=-24, max_value=24, default=0)

    use_era = luigi.parameter.BoolParameter()

    def requires(self):
        return WeatherFillMissingValues(data_dir=self.data_dir, start_date=self.start_date, end_date=self.end_date,
                                        resolution=self.resolution, bounding_box_name=self.bounding_box_name,
                                        fill_method=self.fill_method,
                                        use_era=self.use_era)

    def run(self):
        data = xr.open_dataset(self.input().path)

        logger.debug('Integrating rain')
        data = self.integrate_rain(data)

        if not self.use_era:
            logger.debug('Discarding')
            data = self.discard_offset_measurements(data)

        logger.debug('Computing wind')
        data = self.compute_wind_speed(data)
        logger.debug('Computing vpd')
        data = self.compute_vpd(data)

        logger.debug('Saving data')
        with self.output().temporary_path() as temp_output_path:
            data.to_netcdf(temp_output_path, engine='netcdf4')

    def output(self):
        fn_in, _ = os.path.splitext(os.path.split(self.input().path)[1])
        fn = fn_in + '_grid_%droff' % self.rain_offset + '.nc'

        dest_path = os.path.join(self.data_dir, self.dest_data_dir, fn)

        return luigi.LocalTarget(dest_path)

    @staticmethod
    def compute_vpd(data):
        temp_k = data['temperature']
        rel_humid = data['humidity']

        temp_k = np.array(temp_k, dtype=np.float64)
        rel_humid = np.array(rel_humid, dtype=np.float64)

        logger.debug('Computing VPD')
        """
        temp_c = temp_k - 273.15

        print('temp_c size:', temp_c.size)

        # Saturation Vapor Pressure (es) and Actual Vapor Pressure (ea)
        es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
        print('es size:', es.size)
        ea = rel_humid / 100 * es
        print('ea size:', ea.size)
        vpd = ea - es
        print('vpd size:', vpd.size)
        """

        vpsat = 0.611 * 10 ** ((7.5 * temp_k - 2048.6) / (temp_k - 35.85))
        vp = rel_humid / 100 * vpsat
        vpd = vpsat - vp

        # Ensure it is non-negative
        vpd[vpd < 0] = 0

        data.update({'vpd': (('time', 'y', 'x'), vpd)})

        return data

    def integrate_rain(self, data):
        rain_da = data['precipitation']
        rain_values = np.array(rain_da)

        lag = int(self.rain_offset)
        if self.use_era:
            length = 2
            num_offsets = 2
            upper_off = (lag * num_offsets) + 1
            lower_off = ((length + lag) * num_offsets)
        else:
            length = 4
            num_offsets = len(GFS_OFFSETS)
            upper_off = (lag * num_offsets)
            lower_off = ((length + lag - 1) * num_offsets) + 1

        startup_length = (length + lag) * num_offsets

        integrated_rain = np.empty(rain_values.shape, dtype=rain_values.dtype)
        start = time.time()
        for i in range(0, rain_values.shape[0], num_offsets):
            if i <= upper_off:
                if self.use_era:
                    integrated_rain[i] = np.sum(rain_values[0], axis=0)
                else:
                    integrated_rain[i] = np.sum(rain_values[num_offsets - 1], axis=0)
            elif i < startup_length:
                if self.use_era:
                    integrated_rain[i] = np.sum(rain_values[:i - upper_off:num_offsets], axis=0)
                else:
                    integrated_rain[i] = np.sum(rain_values[num_offsets - 1:i - upper_off:num_offsets], axis=0)
            else:
                integrated_rain[i] = np.sum(rain_values[i - lower_off:i - upper_off:num_offsets], axis=0)

                if (i % (num_offsets * 1000)) == 0:
                    logger.debug('Current rain ind: %d' % i)
                    logger.debug('Val: {:f}, Range: {}, Start: {:d}, End: {:d}'.format(integrated_rain[i, 0, 0], str(
                        rain_values[i - lower_off:i - upper_off:num_offsets, 0, 0]), i - lower_off, i - upper_off))
        if self.use_era:
            for i in range(0, rain_values.shape[0] - 1, num_offsets):
                if i + 2 >= rain_values.shape[0]:
                    integrated_rain[i + 1] = integrated_rain[i]
                else:
                    integrated_rain[i + 1] = (integrated_rain[i] + integrated_rain[i + 2]) / 2
        logger.debug('Time: %f' % (time.time() - start))

        rain_da[:] = integrated_rain

        data = data.rename({'precipitation': 'precipitation_24hr'})

        return data

    @staticmethod
    def discard_offset_measurements(data):
        # offset = pd.to_timedelta(data.offset)
        # non_offset_inds = offset.seconds == 0

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
        times = np.array(data.time)[non_offset_slice]
        lat = np.array(data.lat)
        lon = np.array(data.lon)

        logger.debug('Building dataset')
        ds_new = xr.Dataset(data_arrays, coords={'lat': (['y'], lat), 'lon': (['x'], lon),
                                                 'time': times}, attrs=data.attrs)

        for measurement in ds_new.data_vars.keys():
            ds_new[measurement].encoding = encodings[measurement]

        return ds_new

    @staticmethod
    def compute_wind_speed(data):
        u_wind = data['u_wind_component']
        v_wind = data['v_wind_component']

        logger.debug('Loading wind data')
        u_wind_speed = np.array(u_wind, dtype=np.float64)
        v_wind_speed = np.array(v_wind, dtype=np.float64)

        logger.debug('Computing wind magnitude')
        squared_values = np.square(u_wind_speed) + np.square(v_wind_speed)

        # for v in squared_values:
        #    t = np.sqrt(v)
        #    """
        #    try:
        #        t = np.sqrt(v)
        #    except:
        #        print('Invalid', str(v))
        #    """

        wind_speed = np.sqrt(squared_values).astype(u_wind.dtype)

        logger.debug('Updating wind values')
        u_wind[:] = wind_speed

        logger.debug('Dropping wind values')
        data = data.drop(['v_wind_component'])

        logger.debug('Renaming wind values')
        data = data.rename({'u_wind_component': 'wind_speed'})

        return data
