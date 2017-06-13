import pandas as pd
import numpy as np
import cPickle
from geometry.grid_conversion import get_latlon_xy_fxns, ak_bb
from geometry.get_xys import append_xy
from util.daymonth import monthday2day, day2monthday
from geometry.grid_conversion import get_gfs_val, get_gfs_for_region
from prediction.fire_clustering import cluster_fires
from scipy.spatial import ConvexHull
import sys


def add_daymonth(df):
    days = map(lambda x,y,z: monthday2day(x,y,leapyear=(z%4)), df.month, df.day, df.year)
    df.loc[:,'dayofyear'] = days
    return df


############## COMPUTING FEAT DF FXNS #########

def compute_feat_df(year, fire_df, clusts, gfs_dict_dict):
    """ Get a DataFrame to make active fire prediction easy
    :param year: Year we want to look at
    :param fire_df: DataFrame of active fires. Should contain fields day, month, x, and y
    :param clusts: Cluster assignments for each detection
    :param gfs_dict_dict: Dict of dicts, each inner dict representing a GFS (weather) layer
    :return: a DataFrame for prediction, with fields fire id, day, day_cent, n_det, n_det_cum, hull_size, hull_size_cum,
                gfs...  where we have as many gfs fields as the len zof gfs_dict_dict
    """
    detections = fire_df[fire_df.year == year]
    N = len(detections)
    clust_vals = np.unique(clusts)

    df_dict = dict()
    df_dict['fire_id'] = []
    df_dict['day'] = []
    df_dict['day_cent'] = []
    df_dict['n_det'] = []
    df_dict['n_det_cum'] = []
    #df_dict['hull_size'] = []
    #df_dict['hull_size_cum'] = []
    df_dict['lat'] = []
    df_dict['lon'] = []
    df_dict['x'] = []
    df_dict['y'] = []
    for name in gfs_dict_dict.keys():
        df_dict[name] = []

    for clust in clust_vals:
        clust_dets = detections[clusts == clust]
        days = clust_dets.dayofyear
        min_day = np.min(days)
        max_day = np.max(days)
        center_lat = np.mean(clust_dets.lat)
        center_lon = np.mean(clust_dets.long)
        center_x = np.mean(clust_dets.x)
        center_y = np.mean(clust_dets.y)
        for day in xrange(min_day, max_day+1):
            # We'll have exactly one entry in our DataFrame for this cluster on this day
            df_dict['lat'].append(center_lat)
            df_dict['lon'].append(center_lon)
            df_dict['x'].append(center_x)
            df_dict['y'].append(center_y)
            day_dets = clust_dets[(clust_dets.dayofyear == day)]
            cum_dets = clust_dets[(clust_dets.dayofyear <= day)]
            df_dict['fire_id'].append(clust)
            df_dict['day'].append(day)
            df_dict['day_cent'].append(day - min_day)
            df_dict['n_det'].append(len(day_dets))
            df_dict['n_det_cum'].append(len(cum_dets))
            #if len(day_dets) > 2:
            #    xys = np.column_stack((day_dets.x, day_dets.y))
            #    df_dict['hull_size'].append(ConvexHull(xys).volume)
            #else:
            #    df_dict['hull_size'].append(0.)
            #if len(cum_dets) > 2:
            #    xys_cum = np.column_stack((cum_dets.x, cum_dets.y))
            #    df_dict['hull_size_cum'].append(ConvexHull(xys_cum).volume)
            #else:
            #    df_dict['hull_size_cum'].append(0.)

            month, dayofmonth = day2monthday(day, leapyear=(year%4))
            for name, gfs_dict in gfs_dict_dict.iteritems():
                try:
                    gfs_val = get_gfs_val(center_lat, center_lon, dayofmonth, month, gfs_dict, year)
                    df_dict[name].append(gfs_val)
                except KeyError:
                    df_dict[name].append(np.nan)
                except IndexError:
                    df_dict[name].append(np.nan)

    return pd.DataFrame(df_dict)


def compute_global_feat_df(fire_df, gfs_dict_dict, clust_thresh=10):
    """ Get a DataFrame to make active fire prediction easy
    :param year: Year we want to look at
    :param fire_df: DataFrame of active fires. Should contain fields day, month, x, and y
    :param clusts: Cluster assignments for each detection
    :param gfs_dict_dict: Dict of dicts, each inner dict representing a GFS (weather) layer
    :return: a DataFrame for prediction, with fields fire id, day, day_cent, n_det, n_det_cum, hull_size, hull_size_cum,
                gfs...  where we have as many gfs fields as the len zof gfs_dict_dict
    """
    years = fire_df.year.unique()
    df_dict = dict()
    df_dict['dayofyear'] = []
    df_dict['day'] = []
    df_dict['month'] = []
    df_dict['year'] = []
    df_dict['n_det'] = []
    df_dict['n_clusters'] = []
    for name in gfs_dict_dict.keys():
        df_dict[name] = []

    year = min(years)
    annual_fires = fire_df[fire_df.year == year]
    dayofyear = np.min(annual_fires.dayofyear)
    max_day = np.max(annual_fires.dayofyear)
    month, day = day2monthday(dayofyear, leapyear=not(year % 4))
    while year <= max(years):
        df_dict['day'].append(day)
        df_dict['month'].append(month)
        df_dict['year'].append(year)
        df_dict['dayofyear'].append(dayofyear)
        today_fires = annual_fires[(annual_fires.day == day) & (fire_df.month == month)]
        df_dict['n_det'].append(len(today_fires))
        if len(today_fires):
            n_clusts, _ = cluster_fires(today_fires, clust_thresh, return_df=False)
        else:
            n_clusts = 0
        df_dict['n_clusters'].append(n_clusts)
        for name, gfs_dict in gfs_dict_dict.iteritems():
            try:
                mean_gfs = np.mean(get_gfs_for_region(day, month, year, gfs_dict))  # default bb is ak_inland_bb
                df_dict[name].append(mean_gfs)
            except KeyError:
                df_dict[name].append(np.nan)
            except IndexError:
                df_dict[name].append(np.nan)

        dayofyear += 1
        if dayofyear >= max_day:
            year += 1
            annual_fires = fire_df[fire_df.year == year]
            dayofyear = np.min(annual_fires.dayofyear)
            max_day = np.max(annual_fires.dayofyear)
            month, day = day2monthday(dayofyear, leapyear=not(year % 4))
        else:
            month,day = day2monthday(dayofyear, (year % 4) == 0)

    return pd.DataFrame(df_dict)


def compute_gridded_feat_df(fire_df, gfs_dict_dict, grid_len=4, bb=ak_bb):
    """ Compute a DataFrame for predictive modeling where we break up the space into a grid
    :param fire_df: DataFrame with MODIS fire data
    :param gfs_dict_dict: A dict mapping names of GFS features to dicts representing that feature
    :param cell_size: size, in km, of each grid cell
    :param bb: bounding box for space
    :return: the newly created DataFrame and a mapping from grid cell to lat,lon
    """
    if "x" not in fire_df:
        fire_df = append_xy(fire_df, bb)
    years = fire_df.year.unique()
    df_dict = dict()
    df_dict['dayofyear'] = []
    df_dict['year'] = []
    df_dict['n_det'] = []
    df_dict['grid_i'] = []
    df_dict['grid_j'] = []
    for name in gfs_dict_dict.keys():
        df_dict[name] = []

    year = min(years)
    annual_fires = fire_df[fire_df.year == year]
    dayofyear = np.min(annual_fires.dayofyear)
    max_day = np.max(annual_fires.dayofyear)
    month, day = day2monthday(dayofyear, leapyear=not(year % 4))
    while year <= max(years):
        df_dict['day'].append(day)
        df_dict['month'].append(month)
        df_dict['year'].append(year)
        df_dict['dayofyear'].append(dayofyear)
        today_fires = annual_fires[(annual_fires.day == day) & (fire_df.month == month)]
        df_dict['n_det'].append(len(today_fires))
#        else:
#            n_clusts = 0
#        df_dict['n_clusters'].append(n_clusts)
        for name, gfs_dict in gfs_dict_dict.iteritems():
            try:
                mean_gfs = np.mean(get_gfs_for_region(day, month, year, gfs_dict))  # default bb is ak_inland_bb
                df_dict[name].append(mean_gfs)
            except KeyError:
                df_dict[name].append(np.nan)
            except IndexError:
                df_dict[name].append(np.nan)

        dayofyear += 1
        if dayofyear >= max_day:
            year += 1
            annual_fires = fire_df[fire_df.year == year]
            dayofyear = np.min(annual_fires.dayofyear)
            max_day = np.max(annual_fires.dayofyear)
            month, day = day2monthday(dayofyear, leapyear=not(year % 4))
        else:
            month,day = day2monthday(dayofyear, (year % 4) == 0)

    return pd.DataFrame(df_dict)


########### INTERFACE FXNS #############

def get_feat_df(year, outfile=None, fire_df_loc='/extra/zbutler0/data/west_coast.pkl',
                gfs_locs=('/extra/zbutler0/data/temp_dict.pkl', '/extra/zbutler0/data/hum_dict.pkl',
                          '/extra/zbutler0/data/vpd_dict.pkl'),
                gfs_names=('temp','humidity','vpd'), clust_thresh=20):
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

    n_fires, fires = cluster_fires(fire_df[fire_df.year == year], clust_thresh)
    feat_df = compute_feat_df(year, fire_df, fires, gfs_dict_dict)

    if outfile:
        with open(outfile,'w') as fout:
            cPickle.dump(feat_df, fout, cPickle.HIGHEST_PROTOCOL)
    return feat_df


def get_multiple_feat_dfs(first_year, last_year, base_file_name):
    for year in xrange(first_year, last_year+1):
        get_feat_df(year, base_file_name + "_%d.pkl" % year)


def get_global_df(outfile=None, fire_df_loc='/extra/zbutler0/data/ak_fires.pkl',
                gfs_locs=('/extra/zbutler0/data/temp_dict.pkl', '/extra/zbutler0/data/hum_dict.pkl',
                          '/extra/zbutler0/data/vpd_dict.pkl'),
                gfs_names=('temp','humidity','vpd'), clust_thresh=20):
    print "debug 6"
    with open(fire_df_loc) as ffire:
        fire_df = cPickle.load(ffire)
    if "dayofyear" not in fire_df:
        fire_df = add_daymonth(fire_df)
    gfs_dict_dict = dict()
    for loc,name in zip(gfs_locs, gfs_names):
        with open(loc) as fpkl:
            gfs_dict_dict[name] = cPickle.load(fpkl)

    feat_df = compute_global_feat_df(fire_df, gfs_dict_dict, clust_thresh)
    if outfile:
        with open(outfile, 'w') as fout:
            cPickle.dump(feat_df, fout)
    return feat_df


if __name__ == "__main__":
    get_multiple_feat_dfs(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])

