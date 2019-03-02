import datetime as dt
import itertools
import os
import sys

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

YEARS = np.arange(2007, 2016 + 1)
TARGET_BB = [(71, -165), (55, -138)]
SCALE = 0.5


def convert_spatial(data, data_lat, data_lon, target_bb, scale):
    time = np.arange(data.shape[0])
    # lat is made negative so it is strictly increasing (rather than decreasing)
    rgi = RegularGridInterpolator([time, data_lat * -1, data_lon], data)

    # Define points to sample
    time_sample = time

    lat_sample = np.arange(target_bb[0][0] * -1, target_bb[1][0] * -1 + scale, scale)
    lon_sample = np.arange(target_bb[0][1], target_bb[1][1] + scale, scale)

    sample_points = np.array(list(itertools.product(time_sample, lat_sample, lon_sample)))
    sampled = rgi(sample_points).reshape((len(time_sample), len(lat_sample), len(lon_sample)))

    return sampled


def convert_era(src_path, src_path_precip, dest_path):
    print(src_path, dest_path)
    ds = xr.open_dataset(src_path)
    lat = np.array(ds.latitude)
    lon = np.array(ds.longitude) - 360

    data_arrays = {}

    temp = np.array(ds['t2m'])
    temp = convert_spatial(temp, lat, lon, TARGET_BB, SCALE)
    data_arrays['temperature'] = temp

    u_wind = np.array(ds['u10'])
    u_wind = convert_spatial(u_wind, lat, lon, TARGET_BB, SCALE)
    data_arrays['u_wind_component'] = u_wind

    v_wind = np.array(ds['v10'])
    v_wind = convert_spatial(v_wind, lat, lon, TARGET_BB, SCALE)
    data_arrays['v_wind_component'] = v_wind

    temp_c = temp - 273.15

    temp_dew = np.array(ds['d2m'])
    temp_dew = convert_spatial(temp_dew, lat, lon, TARGET_BB, SCALE)

    temp_dew_c = temp_dew - 273.15
    rel_humidity = 100 * (
            np.exp((17.625 * temp_dew_c) / (243.04 + temp_dew_c)) / np.exp((17.625 * temp_c) / (243.04 + temp_c)))

    data_arrays['humidity'] = rel_humidity

    ds.close()

    # Precipitation
    ds_precip = xr.open_dataset(src_path_precip)

    precip = np.array(ds_precip['tp'])
    precip = convert_spatial(precip, lat, lon, TARGET_BB, SCALE)
    precip[precip < 0] = 0

    precip_full_time = np.full_like(rel_humidity, np.nan)
    precip_full_time[::2] = precip
    data_arrays['precipitation'] = precip_full_time

    data_arrays = {k: (['time', 'y', 'x'], v) for k, v in data_arrays.items()}
    file_name = '/lv_scratch/scratch/graffc0/fire_prediction/data/interim/gfs/region/4/gfs_4_alaska_20070101_20161231' \
                '.nc '
    with xr.open_dataset(file_name) as temp_in:
        lats = np.array(temp_in.lat)
        lons = np.array(temp_in.lon)
    dates = ds.time
    offsets = np.array([dt.timedelta(hours=0) for _ in dates])

    ds_new = xr.Dataset(data_arrays, coords={'lat': (['y'], lats), 'lon': (['x'], lons),
                                             'time': dates, 'offset': (['time'], offsets)})

    # Save
    encoding = {k: {'zlib': True, 'complevel': 1} for k in ds_new.data_vars.keys()}
    ds_new.to_netcdf(dest_path, engine='netcdf4', encoding=encoding)


if __name__ == '__main__':
    print(sys.argv[1], sys.argv[2])
    dest_name = 'eraanl_default_v1_alaska_%d_0101_1231.hdf5'
    for year in YEARS:
        src_path = os.path.join(sys.argv[1], str(year) + '.nc')
        src_path_precip = os.path.join(sys.argv[1], str(year) + '_precip.nc')
        dest_path = os.path.join(sys.argv[2], dest_name % year)
        convert_era(src_path, src_path_precip, dest_path)
