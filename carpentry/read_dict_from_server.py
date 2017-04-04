import cPickle
import numpy as np
from ftplib import FTP
import pygrib
import os
import ftplib
import sys

input_server = "zbutler@datalab-11.ics.uci.edu:/extra/zbutler0/data/gfs/"  # Where we will store raw data


def get_dict_from_server(out_fi, temp_fi, tensor_type='temp', year=2013, local=True):
    res_dict = dict()

    days_arr = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Days in a month
    day = 1
    month = 1
    day_of_year = 0
    first_grib = 1
    min_val = np.inf
    max_val = -np.inf

    while month < 13:  # Get data for every day of the year
        ymd_str = "%d%.2d%.2d" % (year, month, day)
        try:
            if local:
                grib_fi = input_server.split(":")[-1] + ymd_str + ".grb"
            else:
                os.system("scp %s/%s.grb %s" % (input_server, ymd_str, temp_fi))
                grib_fi = temp_fi
            # Now, pull the temperature data to store locally
            grbs = pygrib.open(grib_fi)
        except IOError:
            print "couldn't find %d/%d" %(month, day)
            if day == days_arr[month-1]:
                day = 1
                month += 1
            else:
                day += 1
            day_of_year += 1
            continue

        if tensor_type.startswith('temp'):
            layer = grbs.select(name='Temperature',typeOfLevel='surface')[0]
        elif tensor_type.startswith('hum'):
            layer = grbs.select(name='Relative humidity', level=0, typeOfLevel='unknown',
                                typeOfFirstFixedSurface='200')[0]
        else:
            raise ValueError("Unknown tensor type")
        res_dict[(month, day)] = layer.values
        if first_grib:
            lats,lons = layer.latlons()
            res_dict['lats'] = lats
            res_dict['lons'] = lons
            first_grib = 0
        mn = np.min(layer.values)
        mx = np.max(layer.values)
        if mn < min_val:
            min_val = mn
        if mx > max_val:
            max_val = mx

        print "Finished processing month %d/%d" % (month, day)
        if day == days_arr[month-1]:
            day = 1
            month += 1
        else:
            day += 1
        day_of_year += 1

    res_dict['min'] = min_val
    res_dict['max'] = max_val

    with open(out_fi,"w") as fout:
        cPickle.dump(res_dict, fout, cPickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    get_dict_from_server(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))