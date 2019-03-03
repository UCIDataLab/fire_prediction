import datetime as dtime
import os

import luigi
import parse
import xarray as xr

from src.evaluation.setup_data_structs import build_x_grid_nw
from .fire_pipeline import FireGridGeneration
from .pipeline_helper import check_date_str_spanning
from .pipeline_params import GFS_RESOLUTIONS, WEATHER_FILL_METHOD, REGION_BOUNDING_BOXES
from .weather_pipeline import WeatherGridGeneration


class GridDatasetGeneration(luigi.Task):
    """ Build dataset for training/testing grid models. """
    data_dir: str = luigi.parameter.Parameter()
    dest_data_dir: str = luigi.parameter.Parameter(default='processed/grid/')

    start_date: dtime.date = luigi.parameter.DateParameter()
    end_date: dtime.date = luigi.parameter.DateParameter()

    resolution = luigi.parameter.ChoiceParameter(choices=GFS_RESOLUTIONS)
    bounding_box_name = luigi.parameter.ChoiceParameter(choices=REGION_BOUNDING_BOXES.keys())

    fill_method = luigi.parameter.ChoiceParameter(choices=WEATHER_FILL_METHOD)

    forecast_horizon = luigi.parameter.NumericalParameter(var_type=int, min_value=0, max_value=10)
    rain_offset = luigi.parameter.NumericalParameter(var_type=int, min_value=-24, max_value=24, default=0)

    use_era = luigi.parameter.BoolParameter()

    # num_y_memory = luigi.parameter.NumericalParameter(var_type=int, min_value=0, max_value=100)
    # num_x_memory = luigi.parameter.NumericalParameter(var_type=int, min_value=0, max_value=100)
    # active_check_days = luigi.parameter.NumericalParameter(var_type=int, min_value=1, max_value=20)

    def requires(self):
        weather_task = WeatherGridGeneration(data_dir=self.data_dir, start_date=self.start_date,
                                             end_date=self.end_date, resolution=self.resolution,
                                             bounding_box_name=self.bounding_box_name,
                                             fill_method=self.fill_method, rain_offset=self.rain_offset,
                                             use_era=self.use_era)
        fire_task = FireGridGeneration(data_dir=self.data_dir, start_date=self.start_date, end_date=self.end_date,
                                       bounding_box_sel_name=self.bounding_box_name)

        return {'weather': weather_task, 'fire': fire_task}

    def run(self):
        # Load inputs
        weather_data = xr.open_dataset(self.input()['weather'].path)
        fire_data = xr.open_dataset(self.input()['fire'].path)
        land_cover_data = None

        # Build data
        t_k_arr = [self.forecast_horizon]
        years = range(self.start_date.year, self.end_date.year + 1)
        ds = build_x_grid_nw(weather_data, fire_data, land_cover_data, t_k_arr, years=years)

        ds = ds[self.forecast_horizon]

        # Save outputs
        with self.output().temporary_path() as temp_output_path:
            ds.to_netcdf(temp_output_path, engine='netcdf4')

    def output(self):
        weather_fn, _ = os.path.splitext(os.path.split(self.input()['weather'].path)[1])
        fire_fn, _ = os.path.splitext(os.path.split(self.input()['fire'].path)[1])

        # ds_info = '%dy_%dx_%dk' % (self.num_y_memory, self.num_x_memory, self.forecast_horizon)
        # fn = '_'.join(['grid_ds', weather_fn, fire_fn, ds_info]) + '.nc'

        # fn = 'grid_ds_gfs_%s_modis_%s_%s_%s_%s_%dy_%dx_%da_%dk.nc' % (self.resolution, self.bounding_box_name,
        #        self.start_date, self.end_date, self.fill_method, self.num_y_memory, self.num_x_memory, 
        #        self.active_check_days, self.forecast_horizon) 
        if self.use_era:
            fn = 'grid_ds_era_{}_modis_{}_{}_{}_{}_{:d}roff_{:d}k.nc'.format(
                self.resolution, self.bounding_box_name, self.start_date, self.end_date, self.fill_method,
                self.rain_offset, self.forecast_horizon)
        else:
            fn = 'grid_ds_gfs_{}_modis_{}_{}_{}_{}_{:d}roff_{:d}k.nc'.format(
                self.resolution, self.bounding_box_name, self.start_date, self.end_date, self.fill_method,
                self.rain_offset, self.forecast_horizon)
        file_path = os.path.join(self.data_dir, self.dest_data_dir, fn)

        # file_path = check_spanning_file(file_path, grid_dataset_span_check_func)

        return luigi.LocalTarget(file_path)


"""
def grid_dataset_span_check_func(test_fn, cur_fn):
    fmt = 'grid_ds_gfs_{resolution}_modis_{bb_name}_{start_date}_{end_date}_{fill}_{y_mem}y_{x_mem}x_{active}a_{' \
          'horizon}k.nc '

    test_p = parse.parse(fmt, test_fn)
    cur_p = parse.parse(fmt, cur_fn)

    check = True
    check = check and (test_p['resolution'] == cur_p['resolution'])
    check = check and (test_p['bb_name'] == cur_p['bb_name'])
    check = check and (test_p['fill'] == cur_p['fill'])

    check = check and (int(test_p['y_mem']) >= int(cur_p['y_mem']))
    check = check and (int(test_p['x_mem']) >= int(cur_p['x_mem']))
    check = check and (int(test_p['active']) >= int(cur_p['active']))
    file_path = os.path.join(self.data_dir, self.dest_data_dir, fn)

    # file_path = check_spanning_file(file_path, grid_dataset_span_check_func)

    return luigi.LocalTarget(file_path)
"""


def grid_dataset_span_check_func(test_fn, cur_fn):
    fmt = 'grid_ds_gfs_{resolution}_modis_{bb_name}_{start_date}_{end_date}_{fill}_{y_mem}y_{x_mem}x_{active}a_{' \
          'horizon}k.nc '

    test_p = parse.parse(fmt, test_fn)
    cur_p = parse.parse(fmt, cur_fn)

    check = True
    check = check and (test_p['resolution'] == cur_p['resolution'])
    check = check and (test_p['bb_name'] == cur_p['bb_name'])
    check = check and (test_p['fill'] == cur_p['fill'])

    check = check and (int(test_p['y_mem']) >= int(cur_p['y_mem']))
    check = check and (int(test_p['x_mem']) >= int(cur_p['x_mem']))
    check = check and (int(test_p['active']) >= int(cur_p['active']))

    check = check and (int(test_p['horizon']) == int(cur_p['horizon']))

    # Exit early if files don't match
    if not check:
        return check

    # Check dates
    date_fmt = '%Y%m%d'
    check = check and check_date_str_spanning(test_p['start_date'], test_p['end_date'],
                                              cur_p['start_date'], cur_p['end_date'], date_fmt)

    return check
