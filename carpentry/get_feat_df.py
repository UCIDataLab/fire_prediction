import pandas as pd
import numpy as np
import cPickle
from geometry.grid_conversion import get_latlon_xy_fxns, ak_bb
from util.daymonth import monthday2day, day2monthday
from geometry.grid_conversion import get_gfs_val
from prediction.fire_clustering import cluster_fires
from scipy.spatial import ConvexHull


def add_daymonth(df):
    days = map(monthday2day, df.month, df.day)
    df.loc[:,'dayofyear'] = days
    return df


def compute_feat_df(year, fire_df, clusts, gfs_dict_dict):
    """ Get a DataFrame to make active fire prediction easy
    :param year: Year we want to look at
    :param fire_df: DataFrame of active fires. Should contain fields day, month, x, and y
    :param clusts: Cluster assignments for each detection
    :param gfs_dict_dict: Dict of dicts, each inner dict representing a GFS (weather) layer
    :return: a DataFrame for prediction, with fields fire id, day, day_cent, n_det, n_det_cum, hull_size, hull_size_cum,
                gfs...  where we have as many gfs fields as the len of gfs_dict_dict
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
    df_dict['hull_size'] = []
    df_dict['hull_size_cum'] = []
    df_dict['lat'] = []
    df_dict['lon'] = []
    for name in gfs_dict_dict.keys():
        df_dict[name] = []

    for clust in clust_vals:
        clust_dets = detections[clusts == clust]
        days = clust_dets.dayofyear
        min_day = np.min(days)
        max_day = np.max(days)
        center_lat = np.mean(clust_dets.lat)
        center_lon = np.mean(clust_dets.long)
        for day in xrange(min_day, max_day+1):
            # We'll have exactly one entry in our DataFrame for this cluster on this day
            df_dict['lat'].append(center_lat)
            df_dict['lon'].append(center_lon)
            day_dets = clust_dets[(clust_dets.dayofyear == day)]
            cum_dets = clust_dets[(clust_dets.dayofyear <= day)]
            df_dict['fire_id'].append(clust)
            df_dict['day'].append(day)
            df_dict['day_cent'].append(day - min_day)
            df_dict['n_det'].append(len(day_dets))
            df_dict['n_det_cum'].append(len(cum_dets))
            if len(day_dets) > 2:
                xys = np.column_stack((day_dets.x, day_dets.y))
                df_dict['hull_size'].append(ConvexHull(xys).volume)
            else:
                df_dict['hull_size'].append(0.)
            if len(cum_dets) > 2:
                xys_cum = np.column_stack((cum_dets.x, cum_dets.y))
                df_dict['hull_size_cum'].append(ConvexHull(xys_cum).volume)
            else:
                df_dict['hull_size_cum'].append(0.)

            for name, gfs_dict in gfs_dict_dict.iteritems():
                gfs_val = get_gfs_val(center_lat, center_lon, day, gfs_dict)
                df_dict[name].append(gfs_val)

    return pd.DataFrame(df_dict)


def get_feat_df(year, outfile=None, fire_df_loc='data/ak_fires.pkl',
                gfs_locs=('data/temp_dict.pkl', 'data/hum_dict.pkl', 'data/vpd_dict.pkl'),
                gfs_names=('temp','humidity','vpd'), clust_thresh=10):
    with open(fire_df_loc) as ffire:
        fire_df = cPickle.load(ffire)
    if "dayofmonth" not in fire_df:
        fire_df = add_daymonth(fire_df)

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


if __name__ == "__main__":
    get_feat_df(2013, "data/feat_df_2013.pkl")

