""" Helper functions and classes for ETL pipeline. """

import logging
import os

import luigi
import numpy as np
from helper.date_util import create_true_dates

logger = logging.getLogger('pipeline')
logger.setLevel(logging.DEBUG)


def change_data_dir_path(server_data_dir, local_data_dir, path):
    path = path.split(server_data_dir)[1]
    return os.path.join(local_data_dir, path.lstrip('/'))


def check_spanning_file(dest_path, span_check_func):
    """ 
    Check if the target specified by dest_path is spanned by an existing target in the same directory. 

    Spanning is deterimined by the provided span_check_func.
    """
    dir_path, fn = os.path.split(dest_path)

    # If dest directory doesn't exist, exit early
    if not os.path.exists(dir_path):
        return dest_path

    files = os.listdir(dir_path)

    # Check every file in dest_dir to see if it spans the intended target
    for f in files:
        try:
            check = span_check_func(f, fn)
        except Exception as e:
            check = False

        if check:
            return os.path.join(dir_path, f)

    return dest_path


def check_date_str_spanning(test_str_start, test_str_end, cur_str_start, cur_str_end, date_fmt):
    test_start_date = dt.datetime.strptime(test_str_start, date_fmt)
    cur_start_date = dt.datetime.strptime(cur_str_start, date_fmt)
    start_check = test_start_date <= cur_start_date

    test_end_date = dt.datetime.strptime(test_str_end, date_fmt)
    cur_end_date = dt.datetime.strptime(cur_str_end, date_fmt)
    end_check = test_end_date >= cur_end_date

    return start_check and end_check


def build_dates_and_latlon_coords(start_date, end_date, resolution, bounding_box, times, offsets,
                                  inclusive_lon=False):
    true_dates, true_offsets = create_true_dates(start_date, end_date, times, offsets)

    grid_increment = 1. if resolution == '3' else .5
    lats, lons = bounding_box.make_grid(grid_increment, grid_increment, inclusive_lon)
    lats, lons = lats[:, 0], lons[0, :]

    return true_dates, true_offsets, lats, lons


def build_data_arrays(true_dates, lats, lons, variables):
    """ Variables are a tuple (name, dtype). """
    logging.debug('Building data arrays')

    data_arrays = {}

    # Use float32 dtype instead of variable's actual data type to allow use of nan's for missing values
    for name, dtype in variables:
        data_arrays[name] = np.full(shape=(len(true_dates), len(lats), len(lons)), fill_value=np.nan,
                                    dtype=np.float32)

    return data_arrays


class TimeTaskMixin(object):
    """
    A mixin that when added to a luigi task, will print out
    the tasks execution time to standard out, when the task is
    finished
    """

    @luigi.Task.event_handler(luigi.Event.PROCESSING_TIME)
    def print_execution_time(self, processing_time):
        print('=== PROCESSING TIME === ' + str(processing_time))


class FtpFile(luigi.ExternalTask):
    server_name = luigi.parameter.Parameter()
    server_username = luigi.parameter.Parameter(significant=False)
    server_password = luigi.parameter.Parameter(significant=False)

    file_path = luigi.parameter.Parameter()

    resources = {'ftp': 1}

    def output(self):
        return luigi.contrib.ftp.RemoteTarget(self.file_path, host=self.server_name, username=self.server_username,
                                              password=self.server_password)
