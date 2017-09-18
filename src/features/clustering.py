"""
Algorithms for clustering data.
"""

import numpy as np
import datetime as dt
import logging
import scipy.spatial.distance as sd
import scipy.sparse as sp

from helper import date_util as du
from helper import df_util as dfu
from helper import distance as dist

CLUST_TYPE_SPATIAL='spatial'
CLUST_TYPE_SPATIAL_TEMPORAL ='spatial_temporal'
CLUST_TYPE_SPATIAL_TEMPORAL_FORWARDS='spatial_temporal_forwards'

def cluster_spatial(data, max_thresh_km):
    n_fires_total = 0

    year_range = dfu.get_year_range(data, 'datetime_utc')

    # Calculate clusters per year
    for year in range(year_range[0], year_range[1]+1):
        logging.debug('Clustering for year %d' % year)

        # TODO: This year comparison should be done in localtime, but will not cause issue within Alaskan fire season
        df_year = data[data.datetime_utc.dt.year==year]

        # Build array of each (lat,lon) pair
        points = np.transpose(np.array((df_year.lat, df_year.lon)))

        # Build distance matrix 
        logging.debug('Building distance matrix')
        condensed_distance_matrix = sd.pdist(points, dist.dist_latlon_spherical)
        distance_matrix = sd.squareform(condensed_distance_matrix)

        # Determine clusters using connected components of threshold matrix
        logging.debug('Thresholding and connected component calculation')
        threshold_matrix = distance_matrix <= max_thresh_km
        n_fires_year, fire_clusters = sp.csgraph.connected_components(threshold_matrix, directed=False)

        fire_cluster_ids = fire_clusters + n_fires_total # Offset cluster ids by total ids already assigned
        data.loc[df_year.index, 'cluster_id'] = fire_cluster_ids

        n_fires_total += n_fires_year
        logging.debug('Found %d unique clusters for year %d' % (n_fires_year, year))

    logging.debug('Found %d unique clusters for all years' % n_fires_total)
    return data


def cluster_spatial_temporal(data, max_thresh_km, max_thresh_days):

    def spatial_temporal_dist(p1, p2):
        spatial_dist = dist.dist_latlon_spherical(p1, p2)
        temporal_dist = abs(p1[2] - p2[2]).days

        return (spatial_dist <= max_thresh_km) & (temporal_dist <= max_thresh_days)

    data = dfu.add_date_local(data)

    n_fires_total = 0

    year_range = dfu.get_year_range(data, 'datetime_utc')

    # Calculate clusters per year
    for year in range(year_range[0], year_range[1]+1):
        logging.debug('Clustering for year %d' % year)

        # TODO: This year comparison should be done in localtime, but will not cause issue within Alaskan fire season
        df_year = data[data.datetime_utc.dt.year==year]

        # Build array of each (lat,lon) pair
        points = np.transpose(np.array((df_year.lat, df_year.lon, df_year.date_local)))

        # Build distance matrix 
        logging.debug('Building distance matrix')
        condensed_distance_matrix = sd.pdist(points, spatial_temporal_dist)
        distance_matrix = sd.squareform(condensed_distance_matrix)

        # Determine clusters using connected components of threshold matrix
        logging.debug('Thresholding and connected component calculation')
        threshold_matrix = distance_matrix
        n_fires_year, fire_clusters = sp.csgraph.connected_components(threshold_matrix, directed=False)

        fire_cluster_ids = fire_clusters + n_fires_total # Offset cluster ids by total ids already assigned
        data.loc[df_year.index, 'cluster_id'] = fire_cluster_ids

        n_fires_total += n_fires_year
        logging.debug('Found %d unique clusters for year %d' % (n_fires_year, year))

    return data


def cluster_spatial_temporal_forwards(data, max_thresh_km, max_thresh_days):
    data = dfu.add_date_local(data)
    
    year_range = dfu.get_year_range(data, 'datetime_utc')
    for year in range(year_range[0], year_range[1]+1):
        prev_points = []
        logging.debug('Clustering for year %d' % year)

        for day in du.daterange(dt.date(year, 1, 1), dt.date(year+1, 1, 1)):
            df_day = data[data.date_local==day]

            # Iterate over each detection for the day
            for row in df_day.itertuples():
                pass

    return data

