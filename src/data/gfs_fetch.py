"""
Fetches GFS (Global Forecasting System) data.
"""

import click
import logging
import os
import sys
from ftplib import FTP
import itertools
from time import time
import cPickle as pickle

alaska_bb = [55, 71, -165, -138]

def is_leap_year(year):
    return year % 4 == 0

def days_per_month(month, is_leap):
    if is_leap:
        month_arr = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        month_arr = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return month_arr[month-1]

def makedirs_safe(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

server_name = "nomads.ncdc.noaa.gov"  # Server from which to pull the data
username = "anonymous"
password = "graffc@uci.edu"
gfs_loc = "GFS/analysis_only/"  # location on server of GFS data
year_range = [2007, 2007]

year_month_dir_fmt = "%d%.2d"
year_month_day_dir_fmt = "%d%.2d%.2d"
grib_file_fmt = "gfsanl_4_%s_%.4d_%.3d.grb2"

times = [0, 600, 1200, 1800]
offsets = [0, 3, 6,]
time_offset_list = [(t,o) for t in times for o in offsets]

class GfsFetch(object):
    def __init__(self, dest_dir, start_year, end_year):
        self.dest_dir = dest_dir
        self.year_range = (start_year, end_year)

    def fetch(self):
        """
        Fetch raw GFS data within year range.
        """
        bad_days = 0
        start_time = time()

        ftp = FTP(server_name)
        ftp.login(username, password)
        ftp.cwd(gfs_loc)

        for year in range(self.year_range[0], self.year_range[1]+1):
            for month in range(1, 13):
                year_month = year_month_dir_fmt % (year, month)

                # Get list of all days in this month on server
                days_in_month_dir = map(lambda x: x.split("/")[-1], ftp.nlst(year_month))

                # Make month dir
                year_month_dir = os.path.join(self.dest_dir, year_month)
                makedirs_safe(year_month_dir)

                for day in range(1, days_per_month(month, is_leap_year(year))+1):
                    start_time_day = time()
                    year_month_day = year_month_day_dir_fmt % (year, month, day)

                    # Check if day not on server
                    if year_month_day not in days_in_month_dir:
                        logging.debug('Failed: year %d month %d day %d not on server' % (year, month, day))
                        bad_days += 1
                        continue

                    logging.debug('Fetching: year %d month %d day %d' % (year, month, day))

                    # Make day dir
                    year_month_day_dir = os.path.join(self.dest_dir, year_month, year_month_day)
                    makedirs_safe(year_month_day_dir)

                    dir_list_with_fluff = ftp.nlst('/'.join([year_month, year_month_day]))
                    grib_dir_list = map(lambda x: x.split('/')[-1], dir_list_with_fluff)

                    # Retrieve each grib file from server and save in day dir
                    todays_grib_files = [grib_file_fmt % (year_month_day, t, offset) for (t, offset) in time_offset_list]
                    for grib_file in todays_grib_files:
                        # Check if grib file not on server
                        if grib_file not in grib_dir_list:
                            logging.debug('(no grib %s)' % grib_file)
                            continue
                        path = os.path.join(year_month, year_month_day, grib_file)
                        command = 'RETR %s' % path
                        local_grib_path = os.path.join(self.dest_dir, path)

                        # Check if grib already downloaded
                        if not os.path.isfile(local_grib_path):
                            logging.debug('Fetching grib %s' % grib_file)
                            with open(local_grib_path, 'w') as ftmp:
                                ftp.retrbinary(command, ftmp.write)

                        continue

                        # Generate pickle file path
                        pkl_file = os.path.splitext(grib_file)[0] + '.pkl'
                        path = os.path.join(year_month, year_month_day, pkl_file)
                        local_pkl_path = os.path.join(self.dest_dir, path)

                        # Check if pickle file already generated
                        if not os.path.isfile(local_pkl_path):
                            logging.debug('Generating pickle %s' % pkl_file)
                            gribs = pygrib.open(local_grib_path)
                            sel = get_default_selections()
                            with open(local_pkl_path, 'wb') as f:
                                data, lats, lons = GribSelector(sel, alaska_bb).select(gribs)
                                pickle.dump({'data': data, 'lats': lats, 'lons': lons}, f, protocol=pickle.HIGHEST_PROTOCOL)

                    logging.info('(%d seconds)' % (time() - start_time_day))
        # Clean-up
        ftp.quit()

        total_time = (time() - start_time) / 60.
        logging.info('Total time: %d minutes' % total_time)
        logging.info('Bad days: %s' % bad_days)


@click.command()
@click.argument('dest_dir', type=click.Path(exists=True))
@click.option('--start', default=2007, type=click.INT)
@click.option('--end', default=2016, type=click.INT)
@click.option('--log', default='INFO')
def main(dest_dir, start, end, log):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging, log.upper()), format=log_fmt)

    logging.info('Storing data in "%s". Range is [%d, %d].' % (dest_dir, start, end))

    logging.info('Starting fetch for GFS')
    GfsFetch(dest_dir, start, end).fetch()
    logging.info('End fetch for GFS')


if __name__=='__main__':
    main()
