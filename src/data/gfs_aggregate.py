"""
Convert extracted GFS data to WeatherRegion.
"""

import click
import os
import numpy as np
import logging
import cPickle as pickle
import datetime
import pytz

from helper import date_util as du
from helper import weather
from base.converter import Converter


year_month_dir_fmt = "%d%.2d"
year_month_day_dir_fmt = "%d%.2d%.2d"
extracted_file_fmt_half_deg = "gfsanl_4_%s_%.4d_%.3d.pkl"
extracted_file_fmt_one_deg = "gfsanl_3_%s_%.4d_%.3d.pkl"

SCALE_HALF_DEG = '4'
SCALE_ONE_DEG = '3'

times = [0, 600, 1200, 1800]
offsets = [0, 3, 6,]
time_offset_list = [(t,o) for t in times for o in offsets]

def_name_conversion_dict = {'Surface air relative humidity': 'humidity', '2 metre relative humidity': 'humidity', 'Relative humidity': 'humidity', '10 metre U wind component': 'U wind component', '10 metre V wind component': 'V wind component'}

def def_name_func(name):
    # Convert name if entry in dict
    name_dict = def_name_conversion_dict
    if name in name_dict: name = name_dict[name]

    return name.lower().replace(' ', '_')


class GFStoWeatherRegionConverter(Converter):
    """
    Combine all extracted GFS files to a single WeatherRegion.
    """
    def __init__(self, year_start, year_end, scale_sel, measurement_name_func=def_name_func):
        self.year_range = (year_start, year_end)
        self.measurement_name_func = measurement_name_func

        # Choose file format based on selected scale
        if scale_sel==SCALE_HALF_DEG:
            self.extracted_file_fmt = extracted_file_fmt_half_deg
        elif scale_sel==SCALE_ONE_DEG:
            self.extracted_file_fmt = extracted_file_fmt_one_deg
        else:
            raise ValueError('Scale selction "%s" is invalid.' % scale_sel)

    def load(self, src_dir):
        self.src_dir = src_dir

        # Find all src files to process
        available_files = self.get_available_files()
        logging.debug('Finished fetching available files list')

        
        available_files_present = [f[0] for f in available_files]
        if not any(available_files_present):
            logging.debug('No files available from source')
            return None

        self.num_dates = len(available_files)

        all_data = {}
        dates = []
        for i, (is_avail, f) in enumerate(available_files):
            logging.debug('Converting %s (is_available=%s) (%d/%d)' % (f, is_avail, i+1, self.num_dates))
            # Record date
            date, offset = self.get_date_from_name(os.path.basename(f))
            dates.append(du.DatetimeMeasurement(date, offset))

            # Append data
            if is_avail:
                with open(f, 'rb') as fin:
                    file_data = pickle.load(fin)

                self.append_data(all_data, file_data, i)

            else:
                file_data = None


        return all_data, dates, offsets

    def transform(self, data):
        all_data, dates, offsets = data

        # Create WeatherRegion and add WeatherCubes for each measurement
        region = weather.WeatherRegion('gfs_alaska')
        for k,v in all_data.iteritems():
            measurement = all_data[k]
            values, units, bb = measurement['values'], measurement['units'], measurement['bounding_box']
            cube = weather.WeatherCube(k, values, units, bb, ['lat', 'lon', 'time'], dates)
            region.add_cube(cube)

        return region

    def save(self, dest_path, data):
        with open(dest_path, 'wb') as fout:
            pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)

    def get_available_files(self):
        """
        Get list of all available files (within year_range) in src_dir.
        """
        available_files = []

        for year in range(self.year_range[0], self.year_range[1]+1):
            for month in range(1, 13):
                year_month = year_month_dir_fmt % (year, month)

                months_in_dir = [d for d in os.listdir(self.src_dir) if os.path.isdir(os.path.join(self.src_dir, d))]

                if year_month not in months_in_dir:
                    logging.debug('Missing Month: year %d month %d not in source' % (year, month))

                try:
                    days_in_month_dir = [d for d in os.listdir(os.path.join(self.src_dir, year_month)) if os.path.isdir(os.path.join(self.src_dir, year_month, d))]
                except:
                    days_in_month_dir = []

                for day in range(1, du.days_per_month(month, du.is_leap_year(year))+1):
                    year_month_day = year_month_day_dir_fmt % (year, month, day)

                    if year_month_day not in days_in_month_dir:
                        logging.debug('Missing Day: year %d month %d day %d not in source' % (year, month, day))

                    try:
                        grib_dir_list = [d for d in os.listdir(os.path.join(self.src_dir, year_month, year_month_day)) if os.path.isfile(os.path.join(self.src_dir, year_month, year_month_day, d))]
                    except:
                        grib_dir_list = []

                    todays_grib_files = [self.extracted_file_fmt % (year_month_day, t, offset) for (t, offset) in time_offset_list]
                    for grib_file in todays_grib_files:
                        path = os.path.join(self.src_dir, year_month, year_month_day, grib_file)

                        # Check if grib file not on server
                        if grib_file not in grib_dir_list:
                            logging.debug('Missing Extracted File: %s not in source' % grib_file)
                            file_info = (False, path)
                        else:
                            file_info = (True, path)

                        available_files.append(file_info)

        return available_files

    def get_date_from_name(self, file_name):
        name = file_name[9:] # strip prefix

        year = int(name[:4])
        month = int(name[4:6])
        day = int(name[6:8])
        hour = int(name[9:11])
        minute = int(name[11:13])
        offset = int(name[14:17])

        return datetime.datetime(year, month, day, hour, minute, tzinfo=pytz.UTC), datetime.timedelta(hours=offset)

    def append_data(self, all_data, file_data, date_ind):
        for k, v in file_data.iteritems():
            name = self.measurement_name_func(k)
            values, units, bb = v['values'], v['units'], v['bounding_box']

            if name not in all_data:
                all_data[name] = {}
                measurement = all_data[name]

                new_value_array = np.empty((values.shape[0], values.shape[1], self.num_dates), dtype=np.float32)
                new_value_array.fill(np.nan)

                measurement['values'] = new_value_array

                measurement['units'] = units
                measurement['bounding_box'] = bb

            measurement = all_data[name]
            measurement['values'][:,:, date_ind] = values

            if measurement['units'] != units: logging.debug('Units %s and %s don\'t match for %s' % (measurement['units'], units, name))
            if str(measurement['bounding_box']) != str(bb): logging.debug('Bounding boxes %s and %s don\'t match for %s' % (measurement['bounding_box'], bb, name))


@click.command()
@click.argument('src_dir', type=click.Path(exists=True))
@click.argument('dest_path')
@click.option('--start', default=2007, type=click.INT)
@click.option('--scale', default='4', type=click.Choice([SCALE_HALF_DEG, SCALE_ONE_DEG]))
@click.option('--end', default=2016, type=click.INT)
@click.option('--log', default='INFO')
def main(src_dir, dest_path, start, end, scale, log):
    """
    Load GFS data and create a weather region.
    """
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging, log.upper()), format=log_fmt)

    # Check that dest_path (excluding file) exists
    #if not os.path.isdir(os.path.dirname(dest_path)):
    #    print 'Destination path does not exist "%s"' % dest_path

    logging.info('Starting GFS extracted to WeatherRegion conversion')
    GFStoWeatherRegionConverter(start, end, scale).convert(src_dir, dest_path)
    logging.info('Finished GFS extracted to WeatherRegion conversion')


if __name__=='__main__':
    main()
