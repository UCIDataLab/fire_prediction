import datetime as dt
import os
import sys
from collections import defaultdict

import numpy as np

from src.evaluation import evaluate_model as evm
from . import date_util as du
from . import loaders as load
from . import weather

REP_DIR = "/home/cagraff/Documents/dev/fire_prediction/"
SRC_DIR = REP_DIR + 'src/'
DATA_DIR = REP_DIR + 'data/'


def date_to_day_of_year(date):
    return date.year, date.timetuple().tm_yday


def get_date_index(weather_data, target_datetime):
    date_ind = np.searchsorted(weather_data.dates, target_datetime, side='left')

    # Check if left or right element is closer
    if date_ind != 0:
        date_ind_left, date_ind_curr = date_ind - 1, date_ind

        dist_left = abs((weather_data.dates[date_ind_left] - target_datetime).total_seconds())
        dist_curr = abs((weather_data.dates[date_ind_curr] - target_datetime).total_seconds())

        if dist_left < dist_curr:
            date_ind = date_ind_left

    return date_ind


def get_weather_variables(weather_values, weather_data, target_datetime, covariates, fill_n_days):
    # Get date index
    date_ind = get_date_index(weather_data, target_datetime)

    for key in covariates:
        data = weather_data[key].values
        val = data[:, :, date_ind]

        if np.any(np.isnan(val)):
            val = fill_missing_value(data, date_ind, fill_n_days)

        weather_values[key].append(val)


def fill_missing_value(data, date_ind, fill_n_days):
    """
    Try to replace with closest prev day in range [1, fill_n_days].                                                

    If no non-nan value is found, replaces with mean of all values at the given lat/lon.                           
    """
    for day_offset in range(1, fill_n_days + 1):
        new_date_ind = date_ind - day_offset

        if new_date_ind < 0:
            break

        val = data[:, :, new_date_ind]

        if not np.any(np.isnan(val)):
            return val

    return np.nanmean(data[:, :, :], axis=2)


def main():
    ignition_cube_src = os.path.join(DATA_DIR, 'interim/modis/fire_cube/fire_ignition_cube_modis_alaska_2007-2016.pkl')
    detection_cube_src = os.path.join(DATA_DIR,
                                      'interim/modis/fire_cube/fire_detection_cube_modis_alaska_2007-2016.pkl')
    weather_proc_region_src = os.path.join(DATA_DIR, 'interim/gfs/weather_proc/weather_proc_gfs_4_alaska_2007-2016.pkl')

    _, Y_detection_c = evm.setup_ignition_data(ignition_cube_src, detection_cube_src)
    Y_detection_c.name = 'num_det'
    weather_proc_region = load.load_pickle(weather_proc_region_src)

    t_k = int(sys.argv[2])
    print('T_k=%d' % t_k)

    values = defaultdict(list)
    for date in Y_detection_c.dates:
        time = 14
        date += du.INC_ONE_DAY * t_k
        # TODO: I think the lon (153) doesn't matter because of the time of day we have  selected 14
        tzinfo = du.TrulyLocalTzInfo(153, du.round_to_nearest_quarter_hour)
        target_datetime = dt.datetime.combine(date, dt.time(time, 0, 0, tzinfo=tzinfo))

        get_weather_variables(values, weather_proc_region, target_datetime, ['temperature', 'humidity', 'wind', 'rain'],
                              2)

    to_flatten = weather.WeatherRegion('flatten')
    for k, v in values.items():
        values[k] = list(np.rollaxis(np.array(v), 0, 3))
        cube = weather.WeatherCube(k, values[k], None, dates=Y_detection_c.dates)
        to_flatten.add_cube(cube)

    # Shift detections by t_k
    det_shift = Y_detection_c.values
    shape = np.shape(det_shift)[:2] + (t_k,)
    det_shift = np.concatenate((det_shift, np.zeros(shape)), axis=2)
    det_shift = det_shift[:, :, t_k:]

    values, keys = zip(*[(to_flatten.cubes[k].values, k) for k in ['temperature', 'humidity', 'wind', 'rain']])
    values = (Y_detection_c.values,) + values + (det_shift,)
    keys = ('num_det',) + keys + ('num_det_target',)
    to_flatten_arr = np.stack(values, axis=3)

    with open(sys.argv[1], 'wb') as f_out:
        header = 'year,day_of_year,row,col,' + ','.join(keys) + '\n'
        f_out.write(header.encode())
        for i, d in enumerate(Y_detection_c.dates):
            year, day_of_year = date_to_day_of_year(d)
            for row in range(33):
                for col in range(55):
                    line = '%d,%d,%d,%d,%d,%f,%f,%f,%f,%d\n' % (
                            (year, day_of_year, row, col) + tuple(to_flatten_arr[row, col, i, :]))
                    f_out.write(line.encode())


if __name__ == '__main__':
    main()
