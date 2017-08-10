import cPickle
import os
import pandas as pd
from util.daymonth import monthday2day
from geometry.grid_conversion import get_latlon_xy_fxns, ak_bb


def append_xy(df, bb):
    """ Add uniform XY coordinates to a specified location in a DataFrame that has latitudes and longitudes
    :param df: DataFrame with columns named "lat" and "long" for latitude and longitude respectively
    :param bb: Bounding box for location over which to form grid
    :return: DataFrame in the same form as the other but with "x" and "y" columns
    """
    latlon2xy, _, shape = get_latlon_xy_fxns(bb)
    valid_fires = df[(df.lat < bb[1]) & (df.lat > bb[0]) &
                     (df.long < bb[3]) & (df.long > bb[2])]
    xys = map(latlon2xy, valid_fires.lat, valid_fires.long)
    xs = map(lambda x: x[0], xys)
    ys = map(lambda x: x[1], xys)
    valid_fires.loc[:,'x'] = xs
    valid_fires.loc[:,'y'] = ys
    return valid_fires


def convert_to_pd_batch(my_dir, outfi=None, beginning=2007, ending=2016, verbose=False):
    """ Take CSV files with MODIS active fire data and convert them to a DataFrame
    :param my_dir: Directory with MODIS files
    :param outfi: Optional file to dump output DataFrame to
    :param beginning: beginning year to use
    :param ending: ending year to use
    :return: DataFrame with MODIS active fire data
    """
    # First, build up lists with each column
    year_list = []
    month_list = []
    day_list = []
    dayofyear_list = []
    hour_list = []
    minute_list = []
    lat_list = []
    lon_list = []
    frp_list = []
    satellite_list = []
    confidence_list = []
    for i,fname in enumerate(os.listdir(my_dir)):
        year = int(fname.split(".")[1][0:4])
        if not beginning <= year <= ending:
            continue
        with open(os.path.join(my_dir,fname)) as fin:
            fin.readline()   # Ignore header
            for line in fin:
                yyyymmdd = line.split()[0]
                year = int(yyyymmdd[0:4])
                month = int(yyyymmdd[4:6])
                day = int(yyyymmdd[6:])
                year_list.append(year)
                month_list.append(month)
                day_list.append(day)
                dayofyear_list.append(monthday2day(month, day, leapyear=(year % 4 == 0)))
                hhmm = line.split()[1]
                hour_list.append(int(hhmm[0:2]))
                minute_list.append(int(hhmm[2:]))
                lat_list.append(float(line.split()[3]))
                lon_list.append(float(line.split()[4]))
                frp_list.append(float(line.split()[8]))
                satellite_list.append(line.split()[9])
                confidence_list.append(float(line.split()[9]) / 100.)
        if verbose:
            print "finished reading file %d" % i

    # Now, make a dictionary that we will then cast to a DataFrame
    pd_dict = dict()
    pd_dict['year'] = year_list
    pd_dict['month'] = month_list
    pd_dict['day'] = day_list
    pd_dict['dayofyear'] = dayofyear_list
    pd_dict['hour'] = hour_list
    pd_dict['minute'] = minute_list
    pd_dict['lat'] = lat_list
    pd_dict['lon'] = lon_list
    pd_dict['frp'] = frp_list
    pd_dict['confidence'] = confidence_list
    pd_dict['satellite'] = satellite_list
    df = pd.DataFrame(pd_dict)
    print "created DataFrame of size %d" %(len(df))
    if outfi:
        with open(outfi,'w') as fout:
            cPickle.dump(df, fout, protocol=cPickle.HIGHEST_PROTOCOL)
    return df


def get_fire_data(year_range, bb, outfi, modis_loc=None, modis_df=None):
    """ Get MODIS active fire detection data and save in pandas DataFrame
    :param year_range: range of years (inclusive) we want data for
    :param bb: bounding box we want data for
    :param outfi: place to store pandas DataFrame
    :param modis_loc: location where MODIS CSVs live
    :return: nothing, but save modis DataFrame to pickle in outfi
    """
    if modis_df is None:
        modis_df = convert_to_pd_batch(modis_loc, outfi=None, beginning=year_range[0], ending=year_range[1])
    modis_df = append_xy(modis_df, bb)
    with open(outfi, 'w') as fout:
        cPickle.dump(modis_df, fout, protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # Get global MODIS data
    #convert_to_pd_batch("mcd14ml", "data/full_modis.pkl")
    # Just get Alaska with grid system
    get_fire_data((2007,2016), ak_bb,  "data/ak_fires.pkl", "mcd14ml")
