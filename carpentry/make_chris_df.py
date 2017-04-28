import pandas as pd
import numpy as np
import cPickle
from geometry.grid_conversion import get_latlon_xy_fxns, ak_bb
from geometry.get_xys import append_xy
from util.daymonth import monthday2day, day2monthday
from geometry.grid_conversion import get_gfs_val
from prediction.fire_clustering import cluster_fires
from scipy.spatial import ConvexHull
import sys


def add_daymonth(df):
    days = map(monthday2day, df.month, df.day)
    df.loc[:,'dayofyear'] = days
    return df


def get_feat_df(year, outfile=None, fire_df_loc='data/ak_fires.pkl',
                gfs_locs=('data/temp_dict.pkl', 'data/hum_dict.pkl', 'data/vpd_dict.pkl'),
                gfs_names=('temp','humidity','vpd'), clust_thresh=10):
    with open(fire_df_loc) as ffire:
        fire_df = cPickle.load(ffire)
    if "dayofyear" not in fire_df:
        fire_df = add_daymonth(fire_df)

    # If no XYs, create them, assuming we're in Alaska
    if "x" not in fire_df:
        fire_df = append_xy(fire_df, ak_bb)

    gfs_dict_dict = dict()
    for loc,name in zip(gfs_locs, gfs_names):
        with open(loc) as fpkl:
            gfs_dict_dict[name] = cPickle.load(fpkl)

    gfs_vecs = dict()
    for name, gfs_dict in gfs_dict_dict.iteritems():
        gfs_vecs[name] = np.zeros(len(fire_df)) + np.nan

    for i,fire_event in enumerate(fire_df.keys()):
        for name, gfs_dict in gfs_dict_dict.iteritems():
            try:
                lat = fire_df.lat[fire_event]
                lon = fire_df.long[fire_event]
                dayofyear = fire_df.dayofyear[fire_event]
                gfs_vecs[name][i] = get_gfs_val(lat, lon, dayofyear, gfs_dict, year)
            except KeyError:
                pass

    for name, vec in gfs_vecs.iteritems():
        fire_df[name] = pd.Series(vec, index=fire_df.index)

    if outfile:
        with open(outfile,'w') as fout:
            cPickle.dump(fire_df, fout, cPickle.HIGHEST_PROTOCOL)
    return fire_df


def get_multiple_feat_dfs(first_year, last_year, base_file_name):
    for year in xrange(first_year, last_year+1):
        get_feat_df(year, base_file_name + "_%d.pkl" % year)


if __name__ == "__main__":
    get_multiple_feat_dfs(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])

