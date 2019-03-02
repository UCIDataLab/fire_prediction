import os
import pygrib
import sys

import h5py
import numpy as np
from data import grib
from pipeline.pipeline_helper import change_data_dir_path
from pipeline.pipeline_params import GFS_RAW_DATA_DIR, GFS_FILTERED_DATA_DIR, REGION_BOUNDING_BOXES

resolution = '3'
data_dir = '/extra/graffc0/fire_prediction/data/'

year_month_dir_fmt = "%d%.2d"
year_month_day_dir_fmt = "%d%.2d%.2d"


def name_change(fn):
    dest_path = change_data_dir_path(os.path.join(data_dir, GFS_RAW_DATA_DIR, resolution),
                                     os.path.join(data_dir, GFS_FILTERED_DATA_DIR, resolution), fn)
    dest_path, _ = os.path.splitext(dest_path)
    dest_path += '_%s_%s' % ('default_v1', 'global')
    dest_path += '.hdf5'

    return dest_path


def get_files_in(path):
    files = []
    for year in range(2007, 2016 + 1):
        for month in range(1, 12 + 1):
            year_month_dir = os.path.join(path, year_month_dir_fmt % (year, month))
            days = os.listdir(year_month_dir)
            for day in days:
                day_dir = os.path.join(year_month_dir, day)
                file_names = os.listdir(day_dir)
                files += list(map(lambda x: os.path.join(day_dir, x), file_names))
    return files


files_in = get_files_in(sys.argv[1])
files_out = list(map(name_change, files_in))

num_files = len(files_in)

print(num_files, files_in[:6], files_out[:6])

for i, (fn_in, fn_out) in enumerate(zip(files_in, files_out)):
    with pygrib.open(fn_in) as fin:
        humidity = grib.GribSelection('humidity', np.float32).add_selection(
            name='Surface air relative humidity').add_selection(name='2 metre relative humidity').add_selection(
            name='Relative humidity', level=2)
        selections = [humidity]
        bounding_box = REGION_BOUNDING_BOXES['global']

        extracted = grib.GribSelector(selections, bounding_box).select(fin)

        if 'humidity' in extracted:
            vals = extracted['humidity']['values']

            with h5py.File(fn_out) as fout:
                del fout['humidity']
                d = fout.create_dataset(name='humidity', data=vals, dtype=vals.dtype, compression='lzf')
                d.attrs['units'] = extracted['humidity']['units']

        else:
            print('Humidity missing in %s' % fn_in)

        if (i % 500) == 0:
            print('%d / %d' % (i + 1, num_files))
