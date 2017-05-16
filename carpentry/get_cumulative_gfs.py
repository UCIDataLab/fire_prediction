from ftplib import FTP
import sys
import pygrib
import cPickle
import numpy as np

server_name = "nomads.ncdc.noaa.gov"  # Server from which to pull the data
gfs_loc = "GFS/analysis_only/"  # location on server of GFS data
host_name = "zbutler@datalab-11.ics.uci.edu"
remote_dir = "/extra/zbutler0/data/gfs/"  # Where we will store raw data
out_temp_arr = "./data/gfs_temp_2015.pkl"  # Place to store temperature tensor
out_hum_arr = "./data/gfs_hum_2015.pkl"  # Place to store temperature tensor


def increment_day(year, month, day):
    if year % 4:  # not leap year
        days_arr = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Days in a month
    else:  # leap year
        days_arr = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Days in a month

    if day == days_arr[month-1]:
        day = 1
        month += 1
        if month == 13:
            month = 1
            year += 1
    else:
        day += 1

    return year, month, day


def get_cumulative_gfs(year_range, out_fi, temp_grb, layer_name="Total Precipitation"):
    """ Get accumulated GFS values across an entire day (4 measurements per day)
    :param year_range: get data from year year_range[0] to year_range[1] (inclusive on both ends)
    :param out_fi: name of file to output dict to
    :param temp_grb: location to store temporary grib file. overwrite it every time we download a new file
    :param layer_name: name of grib layer to get. Raises error if multiple layers share that name
    :return: a dict from (hour, day, month, year) tuples to a layer matrix.
    """
    # Date/time parameters
    hours_arr = ['0000','0600','1200','1800']
    day = 1
    month = 1
    year = year_range[0]
    bad_times = 0
    bad_days = 0

    # grib dictionary parameters
    res_dict = dict()
    max_val = -np.inf
    min_val = np.inf

    # FTP parameters
    ftp = FTP(server_name)
    ftp.login("anonymous", "zbutler@fire_prediction.github")
    ftp.cwd(gfs_loc)

    try:
        while year <= year_range[1]:  # Get data for every day of the year
            ym_str = "%d%.2d" % (year, month)
            ymd_str = "%d%.2d%.2d" % (year, month, day)

            # Check if today even exists on the server
            if ymd_str not in map(lambda x: x.split("/")[-1], ftp.nlst(ym_str)):
                print "month %d day %d not on server" % (month, day)
                bad_days += 1
                bad_times += 1
                year, month, day = increment_day(year, month, day)
                continue
            dir_list_with_fluff = ftp.nlst('/'.join([ym_str, ymd_str]))
            dir_list = map(lambda x: x.split('/')[-1], dir_list_with_fluff)

            # Now check for each hour
            bad_day = 0
            for hour in hours_arr:
                filename = "gfsanl_4_%s_%s_006.grb2" % (ymd_str, hour)
                # pull the file from the FTP server
                if filename in dir_list:
                    with open(temp_grb, "w") as ftmp:
                        ftp.retrbinary("RETR %s/%s/%s" % (ym_str, ymd_str, filename), ftmp.write)
                    print "Found %s" % filename
                else:
                    print "Didn't find %s" % filename
                    bad_times += 1
                    if not bad_day:
                        bad_day = 1
                        bad_days += 1
                    continue

                # We have the file!
                grbs = pygrib.open(temp_grb)
                select = grbs.select(name=layer_name)
                if len(select) != 1:
                    raise ValueError("expected 1 layer of name %s, got %d" % layer_name, len(select))
                layer = select[0]
                if 'lat' not in res_dict:
                    res_dict['lat'], res_dict['lon'] = layer.latlons()
                res_dict[(year, month, day, hour)] = layer.values
                mx = np.max(layer.values)
                mn = np.min(layer.values)
                if mx > max_val:
                    max_val = mx
                if mn < min_val:
                    min_val = mn

    except Exception as e:  # I know...sue me
        print e

    print "%d bad times, %d bad days" % (bad_times, bad_days)
    res_dict['min'] = min_val
    res_dict['max'] = max_val
    with open(out_fi, 'w') as fout:
        cPickle.dump(res_dict, fout, protocol=cPickle.HIGHEST_PROTOCOL)
    ftp.quit()
    return res_dict


if __name__ == "__main__":
    min_year = int(sys.argv[1])
    max_year = int(sys.argv[2])
    if len(sys.argv) >= 5:
        get_cumulative_gfs((min_year, max_year), sys.argv[3], sys.argv[4])
    else:
        get_cumulative_gfs((min_year, max_year), sys.argv[3], sys.argv[4], sys.argv[5])
