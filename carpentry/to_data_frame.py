import cPickle
import os
import pandas as pd
import numpy as np
from raw_to_dict import read_raw_data
from geometry.grid_conversion import get_latlon_xy_fxns


def convert_to_pd_batch(my_dir, outfi=None, beginning=2013, ending=2016):
    year_list = []
    month_list = []
    day_list = []
    hour_list = []
    minute_list = []
    lat_list = []
    long_list = []
    frp_list = []
    confidence_list = []
    for i,fname in enumerate(os.listdir(my_dir)):
        year = int(fname.split(".")[1][0:4])
        if not beginning <= year <= ending:
            continue
        dl = read_raw_data(os.path.join(my_dir,fname))[1:]
        year_list += map(lambda x: x[0], dl)
        month_list += map(lambda x: x[1], dl)
        day_list += map(lambda x: x[2], dl)
        hour_list += map(lambda x: x[3], dl)
        minute_list += map(lambda x: x[4], dl)
        lat_list += map(lambda x: x[5], dl)
        long_list += map(lambda x: x[6], dl)
        frp_list += map(lambda x: x[7], dl)
        confidence_list += map(lambda x: x[8], dl)
        print "finished reading file %d" %(i)
    pd_dict = dict()
    pd_dict['year'] = year_list
    pd_dict['month'] = month_list
    pd_dict['day'] = day_list
    pd_dict['hour'] = hour_list
    pd_dict['minute'] = minute_list
    pd_dict['lat'] = lat_list
    pd_dict['long'] = long_list
    pd_dict['frp'] = frp_list
    pd_dict['confidence'] = confidence_list
    df = pd.DataFrame(pd_dict)
    print "created DataFrame of size %d" %(len(df))
    if outfi:
        with open(outfi,'w') as fout:
            cPickle.dump(df, fout, protocol=cPickle.HIGHEST_PROTOCOL)
    return df


if __name__ == "__main__":
    convert_to_pd_batch("mcd14ml", "modis.pkl")
