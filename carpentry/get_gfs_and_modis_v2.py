import pandas as pd
import numpy as np
import cPickle
import pygrib
import sys
from geometry.grid_conversion import ak_bb
from ftplib import FTP
from util.daymonth import *


server_name = "nomads.ncdc.noaa.gov"  # Server from which to pull the data
gfs_loc = "GFS/analysis_only/"  # location on server of GFS data
host_name = "zbutler@datalab-11.ics.uci.edu"
remote_dir = "/extra/zbutler0/data/gfs/"  # Where we will store raw data


def get_gfs_region(year_range, bb, fields, outfi, tmpfi, timezone='ak'):
    """ Download GFS for a given region and store in a dict of tensors
    :param year_range: range of years (inclusive) we want data for
    :param bb: bounding box or geopandas shape we want data for
    :param fields: list of names of fields we want. Choose from ['temp', 'humidity', 'wind', 'rain']
    :param outfi: location to store eventual data
    :param tmpfi: Temporary place to store GFS data
    :param gfs_loc: Where to find GFS data locally. If it's not there, download it
    :param timezone: string indicating timezone (currently only supports Alaska) TODO: allow any timezone
    :return: nothing, but save tensor dict to outfi in a pickle.
    """
    # Set time
    year = year_range[0]
    month = 1
    day = 1
    bad_days = 0
    surfaceair = True
    first_time = 1

    # Prep return dictionary. Each field will be a list of matrices at first and we will convert it to tensors
    ret_dict = dict()
    for field in fields:
        ret_dict[field] = []
        if field == 'rain':
            ret_dict['valid_bits'] = []
    ret_dict['days'] = []
    # Open FTP connection
    ftp = FTP(server_name)
    ftp.login("anonymous", "zbutler@fire_prediction.github")
    ftp.cwd(gfs_loc)

    while year <= year_range[1]:
        ym_str = "%d%.2d" % (year, month)
        ymd_str = "%d%.2d%.2d" % (year, month, day)
        tomorrow = increment_day(year, month, day)
        ym_tom_str = "%d%.2d" % (tomorrow[0], tomorrow[1])
        ymd_tom_str = "%d%.2d%.2d" % tomorrow

        # Check if today even exists on the server
        if ymd_str not in map(lambda x: x.split("/")[-1], ftp.nlst(ym_str)):
            print "month %d day %d not on server" % (month, day)
            bad_days += 1
            year, month, day = increment_day(year, month, day)
            continue
        dir_list_with_fluff = ftp.nlst('/'.join([ym_str, ymd_str]))
        dir_list = map(lambda x: x.split('/')[-1], dir_list_with_fluff)
        dir_list_with_fluff = ftp.nlst('/'.join([ym_tom_str, ymd_tom_str]))
        tom_dir_list = map(lambda x: x.split('/')[-1], dir_list_with_fluff)

        # INSTANTANEOUS VARIABLES
        today_grbs_file = "gfsanl_4_%s_1800_003.grb2" % ymd_str
        if today_grbs_file not in dir_list:
            bad_days += 1
            year, month, day = increment_day(year, month, day)
            continue
        with open(tmpfi, 'w') as ftmp:
            ftp.retrbinary("RETR %s/%s/%s" % (ym_str, ymd_str, today_grbs_file), ftmp.write)
        today_grbs = pygrib.open(tmpfi)
        for field in fields:
            if field == "temp":
                layer = today_grbs.select(name='Temperature', typeOfLevel='surface')[0].values
            elif field == "humidity":
                if surfaceair:
                    try:
                        layer = today_grbs.select(name='Surface air relative humidity')[0].values
                    except ValueError:
                        surfaceair = False
                if not surfaceair:
                    try:
                        layer = today_grbs.select(name='2 metre relative humidity')[0].values
                    except ValueError:
                        layer = today_grbs.select(name='Relative humidity', level=2)[0].values
            elif field == "wind":
                u_layer = today_grbs.select(name='10 metre U wind component')[0].values
                v_layer = today_grbs.select(name='10 metre V wind component')[0].values
                layer = np.sqrt(u_layer**2 + v_layer**2)
            elif field == "vpd":
                temp_layer = grbs.select(name='Temperature',typeOfLevel='surface')[0]
                temp_vals = temp_layer.values - 273.15  # Convert to celsius
                if surfaceair:
                    hum_vals = today_grbs.select(name='Surface air relative humidity')[0].values
                else:
                    try:
                        hum_vals = today_grbs.select(name='2 metre relative humidity')[0].values
                    except ValueError:
                        hum_vals = today_grbs.select(name='Relative humidity', level=2)[0].values
                svp = .6108 * np.exp(17.27 * temp_vals / (temp_vals + 237.3))
                layer = svp * (1 - (hum_vals / 100.))
            elif field == "rain":
                continue
            else:
                raise ValueError("Invalid field name. Must be one of ('temp', 'humidity', 'wind', 'vpd', and 'rain'")

            if first_time:
                lats, lons = today_grbs.select(name='Temperature', typeOfLevel='surface')[0].latlons()
                gfs_bb_0 = np.where(lats[:,0] < bb[1])[0][0]
                gfs_bb_1 = np.where(lats[:,0] > bb[0])[0][-1]
                gfs_bb_2 = np.where(lons[0,:] > (bb[2] % 360))[0][0]
                gfs_bb_3 = np.where(lons[0,:] < (bb[3] % 360))[0][-1]
                ret_dict['lats'] = lats[gfs_bb_0:gfs_bb_1, gfs_bb_2:gfs_bb_3]
                ret_dict['lons'] = lons[gfs_bb_0:gfs_bb_1, gfs_bb_2:gfs_bb_3]
                first_time = 0
            ret_dict[field].append(layer[gfs_bb_0:gfs_bb_1, gfs_bb_2:gfs_bb_3])

        # CUMULATIVE VALUES (i.e., rain)
        tuples_list = [(ym_str, ymd_str, '1200'), (ym_str, ymd_str, '1800'),
                       (ym_tom_str, ymd_tom_str, '0000'), (ym_tom_str, ymd_tom_str, '0600')]
        filenames = map(lambda x: "%s/%s/gfsanl_4_%s_%s_006.grb2" %(x[0],x[1],x[1],x[2]), tuples_list)
        valid_bits = [0,0,0,0]
        total_rain_layer = np.zeros(ret_dict['lats'].shape)
        for fnum, filename in enumerate(filenames):
            # pull the file from the FTP server
            try:
                with open(tmpfi, "w") as ftmp:
                    ftp.retrbinary("RETR %s" % filename, ftmp.write)
                valid_bits[fnum] = 1
                grbs = pygrib.open(tmpfi)
                vals = grbs.select(name="Total Precipitation")[0].values
                total_rain_layer += vals[gfs_bb_0:gfs_bb_1, gfs_bb_2:gfs_bb_3]
            except Exception:
                pass
        ret_dict['days'].append((year, month, day))
        year, month, day = increment_day(year, month, day)

    with open(outfi,'w') as fout:
        cPickle.dump(ret_dict, fout, protocol=cPickle.HIGHEST_PROTOCOL)


def get_fire_data(year_range, bb, outfi, modis_loc=None):
    """ Get MODIS active fire detection data and save in pandas DataFrame
    :param year_range: range of years (inclusive) we want data for
    :param bb: bounding box or geopandas shape we want data for
    :param outfi: place to store pandas DataFrame
    :param modis_loc: location where MODIS CSV lives (if it doesn't exist, download it)
    :return: nothing, but save modis DataFrame to pickle in outfi
    """
    pass


if __name__ == "__main__":
    year_range = [int(sys.argv[1]), int(sys.argv[2])]
    get_gfs_region(year_range, ak_bb, ['temp', 'humidity', 'vpd', 'wind', 'rain'],
                   sys.argv[3], sys.argv[4])