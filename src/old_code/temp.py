import os
import sys
from ftplib import FTP
import itertools
from time import time

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
grb_file_fmt = "gfsanl_4_%s_%.4d_%.3d.grb2"

times = [0, 600, 1200, 1800]
offsets = [0, 3, 6,]
time_offset_list = [(t,o) for t in times for o in offsets]

def get_gfs_data(dest_dir, min_year, max_year):
    year_range = [min_year, max_year]
    bad_days = 0
    start_time = time()

    ftp = FTP(server_name)
    ftp.login(username, password)
    ftp.cwd(gfs_loc)

    for year in range(year_range[0], year_range[1]+1):
        for month in range(1, 13):
            year_month = year_month_dir_fmt % (year, month)

            # Get list of all days in this month on server
            days_in_month_dir = map(lambda x: x.split("/")[-1], ftp.nlst(year_month))

            # Make month dir
            year_month_dir = os.path.join(dest_dir, year_month)
            makedirs_safe(year_month_dir)

            for day in range(1, days_per_month(month, is_leap_year(year))+1):
                start_time_day = time()
                year_month_day = year_month_day_dir_fmt % (year, month, day)

                # Check if day not on server
                if year_month_day not in days_in_month_dir:
                    print "Failed: year %d month %d day %d not on server" % (year, month, day)
                    bad_days += 1
                    continue

                print "Fetching: year %d month %d day %d" % (year, month, day),

                # Make day dir
                year_month_day_dir = os.path.join(dest_dir, year_month, year_month_day)
                makedirs_safe(year_month_day_dir)

                dir_list_with_fluff = ftp.nlst('/'.join([year_month, year_month_day]))
                grib_dir_list = map(lambda x: x.split('/')[-1], dir_list_with_fluff)

                # Retrieve each grib file from server and save in day dir
                todays_grb_files = [grb_file_fmt % (year_month_day, t, offset) for (t, offset) in time_offset_list]
                for grb_file in todays_grb_files:
                    # Check if grib file not on server
                    if grb_file not in grib_dir_list:
                        print "(no grib %s)" % grb_file,
                        continue
                    path = os.path.join(year_month, year_month_day, grb_file)
                    command = "RETR %s" % path
                    local_path = os.path.join(dest_dir, path)

                    # Check if grib already downloaded
                    if os.path.isfile(local_path):
                        continue

                    with open(local_path, 'w') as ftmp:
                        ftp.retrbinary(command, ftmp.write)

                print "(%d seconds)" % (time() - start_time_day)

    ftp.quit()
    total_time = (time() - start_time) / 60.
    print "Total time: %d minutes" % total_time 
    print "Bad days: %s" % bad_days

if __name__ == "__main__":
    get_gfs_data(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
