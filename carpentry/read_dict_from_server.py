import cPickle
import numpy as np
import pygrib
import os
import sys
from util.daymonth import increment_day, monthday2day

input_server = "zbutler@datalab-11.ics.uci.edu:/extra/zbutler0/data/gfs/"  # Where we will store raw data
local_dir = '/extra/zbutler0/data/gfs/' #"/Users/zbutler/research/fire_prediction/data/gfs/"


def get_dict_from_server(out_fi, temp_fi, tensor_type='temp', firstyear=2013, lastyear=2013, local=True):
    res_dict = dict()

    day = 1
    month = 1
    year = firstyear
    first_grib = 1
    min_val = np.inf
    max_val = -np.inf
    surfaceair = True

    while year <= lastyear:
        ymd_str = "%d%.2d%.2d" % (year, month, day)
        try:
            if local:
                grib_fi = local_dir + ymd_str + ".grb"
            else:
                os.system("scp %s/%s.grb %s" % (input_server, ymd_str, temp_fi))
                grib_fi = temp_fi
            # Now, pull the temperature data to store locally
            grbs = pygrib.open(grib_fi)
        except IOError:
            print "couldn't find %d/%d" %(month, day)
            year, month, day = increment_day(year, month, day)
            continue

        if tensor_type.startswith('temp'):
            layer = grbs.select(name='Temperature',typeOfLevel='surface')[0].values

        elif tensor_type.startswith('hum'):
            if surfaceair:
                try:
                    layer = grbs.select(name='Surface air relative humidity')[0].values
                except ValueError:
                    surfaceair = False
            if not surfaceair:
                try:
                    layer = grbs.select(name='2 metre relative humidity')[0].values
                except ValueError:
                    layer = grbs.select(name='Relative humidity', level=2)[0].values

        elif tensor_type.startswith('vpd'):
            temp_layer = grbs.select(name='Temperature',typeOfLevel='surface')[0]
            temp_vals = temp_layer.values - 273.15  # Convert to celsius
            if surfaceair:
                try:
                    hum_vals = grbs.select(name='Surface air relative humidity')[0].values
                except ValueError:
                    surfaceair = False
            if not surfaceair:
                try:
                    hum_vals = grbs.select(name='2 metre relative humidity')[0].values
                except ValueError:
                    hum_vals = grbs.select(name='Relative humidity', level=2)[0].values
            svp = .6108 * np.exp(17.27 * temp_vals / (temp_vals + 237.3))
            layer = svp * (1 - (hum_vals / 100.))

        elif tensor_type.startswith('wind'):
            u_comp = grbs.select(name="10 metre U wind component")[0].values
            v_comp = grbs.select(name="10 metre V wind component")[0].values
            layer = np.sqrt(u_comp**2 + v_comp**2)

        else:
            raise ValueError("Unknown tensor type")

        res_dict[(month, day, year)] = layer
        if first_grib:
            lats,lons = grbs[1].latlons()
            res_dict['lats'] = lats
            res_dict['lons'] = lons
            first_grib = 0
        mn = np.min(layer)
        mx = np.max(layer)
        if mn < min_val:
            min_val = mn
        if mx > max_val:
            max_val = mx
        print np.min(layer)
        print np.max(layer)
        print "Finished processing month %d/%d" % (month, day)
        year, month, day = increment_day(year, month, day)

    res_dict['min'] = min_val
    res_dict['max'] = max_val

    with open(out_fi,"w") as fout:
        cPickle.dump(res_dict, fout, cPickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    get_dict_from_server(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))