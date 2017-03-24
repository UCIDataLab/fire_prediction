import cPickle
import numpy as np
from ftplib import FTP
import pygrib
import os
import ftplib
import sys

input_server = "zbutler@datalab-11.ics.uci.edu:/extra/zbutler0/data/gfs"  # Where we will store raw data


def get_tensor_from_server(out_fi, temp_fi, tensor_type='temp', year=2013):
    tensor = np.zeros((181, 360, 365), dtype=float)

    days_arr = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Days in a month
    day = 1
    month = 1
    day_of_year = 0

    while month < 13:  # Get data for every day of the year
        ymd_str = "%d%.2d%.2d" % (year, month, day)
        os.system("scp %s/%s.grb %s" % (input_server, ymd_str, temp_fi))

        # Now, pull the temperature data to store locally
        grbs = pygrib.open(temp_fi)
        for layer in grbs:
            if tensor_type.startswith('temp'):
                if layer.name == "Temperature" and layer.level == 0:
                    my_layer = layer
                    break
            elif tensor_type.startswith('hum'):
                if layer.name == "Relative humidity" and layer.typeOfLevelECMF == "entireAtmosphere":
                    my_layer = layer
                    break
            else:
                raise ValueError("Unknown tensor type")
        tensor[:,:,day_of_year] = my_layer.values

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