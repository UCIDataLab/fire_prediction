import numpy as np
import pandas as pd

from data import data
from geometry.fire_clustering import cluster_over_time_with_merging
from geometry.grid_conversion import gfs_to_loc_df

POTENTIAL_GFS_VARS = {"temp", "humidity", "wind", "rain"}


def compute_cluster_feat_df(modis_df, gfs_dict, clust_thresh, rain_del=0):
    """ Produce DataFrame for cluster model
    :param modis_df: active fire DataFrame
    :param gfs_dict: dictionary of weather tensors
    :param clust_thresh: cluster threshold
    :param rain_del: If non-zero, add in a rain delay column (empirical optimal is 2)
    :return: a DataFrame for prediction, with a row for each detection, including weather and cluster ID.
    """
    try:
        cluster_df = data.load_clust_df(clust_thresh=clust_thresh)
        merge_dict = data.load_merge_dict(clust_thresh=clust_thresh)
    except IOError:
        print "WARNING: couldn't find cluster df, creating our own clusters"
        cluster_df, merge_dict = cluster_over_time_with_merging(modis_df, clust_thresh)
    gfs_vars = list(set(gfs_dict.keys()).intersection(POTENTIAL_GFS_VARS))
    if "rain" in POTENTIAL_GFS_VARS and rain_del:
        gfs_vars.append("rain_del_%d" % rain_del)
    ret_df = pd.DataFrame(columns=["cluster", "alt_cluster", "lat_centroid", "lon_centroid", "n_det", "dayofyear", "year"] + gfs_vars)
    for cluster in cluster_df.cluster.unique():
        if np.isnan(cluster):
            continue
        clust_fires = cluster_df[cluster_df.cluster == cluster]
        lat_centroid = np.mean(clust_fires.lat)
        lon_centroid = np.mean(clust_fires.long)
        gfs_df = gfs_to_loc_df(gfs_dict, lat_centroid, lon_centroid, outfi=None)
        # There's probably a better way to do this, but I'm not gonna run this code too often, so
        # for loop it is
        unique_years = clust_fires.year.unique()
        if len(unique_years) != 1:
            raise ValueError("Cluster %d spans %d years, should only span 1" % (cluster, len(unique_years)))
        clust_year = clust_fires.year.unique()[0]
        for dayofyear in xrange(np.min(clust_fires.dayofyear), np.max(clust_fires.dayofyear)+1):
            today_clust_fires = clust_fires[clust_fires.dayofyear == dayofyear]
            row_dict = dict()
            row_dict["cluster"] = cluster
            if cluster in merge_dict:
                row_dict["alt_cluster"] = merge_dict[cluster]
            else:
                row_dict["alt_cluster"] = np.nan
            row_dict["n_det"] = len(today_clust_fires)
            row_dict["lat_centroid"] = lat_centroid
            row_dict["lon_centroid"] = lon_centroid
            row_dict["year"] = clust_year
            row_dict["dayofyear"] = dayofyear
            for var in gfs_vars:
                try:
                    if var.startswith("rain_del"):
                        row_dict[var] = float(gfs_df[(gfs_df.year == clust_year) &
                                                     (gfs_df.dayofyear == (dayofyear-rain_del))]["rain"])
                    else:
                        row_dict[var] = float(gfs_df[(gfs_df.year == clust_year) & (gfs_df.dayofyear == dayofyear)][var])
                except TypeError as e:
                    row_dict[var] = np.nan

            ret_df = ret_df.append(row_dict, ignore_index=True)
    return ret_df
