import cPickle
import numpy as np
from ftplib import FTP
import pygrib
import os
import ftplib
import sys

input_server = "zbutler@datalab-11.ics.uci.edu:/extra/zbutler0/data/gfs/"  # Where we will store raw data


def get_tensor_from_server(out_fi, temp_fi, tensor_type='temp', year=2013, local=True):
    tensor = np.zeros((181, 360, 365), dtype=float)

    days_arr = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Days in a month
    day = 1
    month = 1
    day_of_year = 0

    while month < 13:  # Get data for every day of the year
        ymd_str = "%d%.2d%.2d" % (year, month, day)
        if local:
            grib_fi = input_server.split(":")[-1] + ymd_str + ".grb"
        else:
            os.system("scp %s/%s.grb %s" % (input_server, ymd_str, temp_fi))
            grib_fi = temp_fi

        # Now, pull the temperature data to store locally
        grbs = pygrib.open(grib_fi)

        if tensor_type.startswith('temp'):
            layer = grbs[211]
            if layer.name != "Temperature" or layer.level != 0:
                raise ValueError("Incorrect layer on month %d day %d" % (month, day))
        elif tensor_type.startswith('hum'):
            layer = grbs[234]
            if 'humidity' not in layer.name:
                raise ValueError("Incorrect layer on month %d day %d" % (month, day))
        else:
            raise ValueError("Unknown tensor type")
        tensor[:,:,day_of_year] = layer.values

        print "Finished day %d, month %d" % (day, month)
        if day == days_arr[month-1]:
            day = 1
            month += 1
        else:
            day += 1
        day_of_year += 1

    with open(out_fi,"w") as fout:
        cPickle.dump(tensor, fout, cPickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    get_tensor_from_server(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))