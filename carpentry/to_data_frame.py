import cPickle
import os
import pandas as pd
import numpy as np
from raw_to_dict import read_raw_data


def convert_to_pd_batch(my_dir, outfi=None):
    year_list = []
    month_list = []
    day_list = []
    lat_list = []
    long_list = []
    frp_list = []
    confidence_list = []
    for i,fname in enumerate(os.listdir(my_dir)):
        dl = read_raw_data(os.path.join(my_dir,fname))
        year_list.append(map(lambda x: x[0], dl))
        month_list.append(map(lambda x: x[1], dl))
        day_list.append(map(lambda x: x[2], dl))
        lat_list.append(map(lambda x: x[3], dl))
        long_list.append(map(lambda x: x[4], dl))
        frp_list.append(map(lambda x: x[5], dl))
        confidence_list.append(map(lambda x: x[6], dl))
        print "finished reading file %d of %d" %(i, len(os.listdir(my_dir)))
    pd_dict = dict()
    pd_dict['year'] = year_list
    pd_dict['month'] = month_list
    pd_dict['day'] = day_list
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
