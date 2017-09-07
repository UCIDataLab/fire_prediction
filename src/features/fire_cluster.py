"""
Converts a data frame of fire data to a cluser data frame.
"""
import pyximport; pyximport.install()

import os
import click
import cPickle as pickle
import pandas as pd
import logging
import numpy as np
import scipy.sparse as sp
import scipy.spatial.distance as sd
from datetime import date

from base.converter import Converter
from helper import df_util as dfu
from helper import distance as dist
from helper import date_util as du

class FireDfToClusterConverter(Converter):
    """
    Converts a data frame of fire data to a cluser data frame.
    """
    def __init__(self, cluster_id_path=None, cluster_thresh_km=5., fire_season=((5,14), (8,31))):
        super(FireDfToClusterConverter, self).__init__()

        self.cluster_id_path = cluster_id_path
        self.cluster_thresh_km = cluster_thresh_km
        self.fire_season = fire_season

    def load(self, src_path):
        logging.info('Loading file from %s' % src_path)
        with open(src_path, 'rb') as fin:
            data = pickle.load(fin)

        return data

    def save(self, dest_path, data):
        logging.info('Saving data frame to %s' % dest_path)

        with open(dest_path, 'wb') as fout:
            pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)

    def transform(self, data):
        logging.debug('Applying transforms to data frame')

        # Add local datetime col to determine the day the fire occured in
        df = data.assign(date_local=map(lambda x: du.utc_to_local_time(x[0], x[1], du.round_to_nearest_quarter_hour).date(), zip(data.datetime_utc, data.lon)))

        # Build cluster id data frame
        if not (self.cluster_id_path and os.path.isfile(self.cluster_id_path)):
            logging.debug('Building cluster id data frame')
            fire_season_df = self.filter_fire_season(df)
            cluster_id_df = self.append_cluster_id(fire_season_df)
            if self.cluster_id_path: self.save(self.cluster_id_path, cluster_id_df)
        else:
            cluster_id_df = self.load(self.cluster_id_path)

        cluster_df = self.build_cluster_df(cluster_id_df)

        cluster_df.sort_values('date_local', inplace=True)
        cluster_df.reset_index(drop=True, inplace=True)

        return cluster_df

    def filter_fire_season(self, df):
        logging.debug('Filtering to only include fire season %s' % str(self.fire_season))

        year_range = dfu.get_year_range(df, 'datetime_utc')
        year_dfs = []
        for year in range(year_range[0], year_range[1]+1):
            begin_season = date(year, self.fire_season[0][0], self.fire_season[0][1])
            end_season = date(year, self.fire_season[1][0], self.fire_season[1][1])

            year_dfs.append(df[(df.date_local >= begin_season) & (df.date_local <= end_season)])

        return pd.concat(year_dfs)

    def append_cluster_id(self, data):
        """
        Appends cluster id to each row of data frame.
        """
        logging.debug('Appending cluster ids')
        n_fires_total = 0

        year_range = dfu.get_year_range(data, 'datetime_utc')

        # Calculate clusters per year
        for year in range(year_range[0], year_range[1]+1):
            logging.debug('Clustering for year %d' % year)

            df_year = data[data.datetime_utc.dt.year==year]

            # Build array of each (lat,lon) pair
            points = np.transpose(np.array((df_year.lat, df_year.lon)))

            # Build distance matrix 
            logging.debug('Building distance matrix')
            condensed_distance_matrix = sd.pdist(points, dist.dist_latlon_spherical)
            distance_matrix = sd.squareform(condensed_distance_matrix)

            # Determine clusters using connected components of threshold matrix
            logging.debug('Tresholding and connected component calculation')
            treshold_matrix = distance_matrix < self.cluster_thresh_km
            n_fires_year, fire_clusters = sp.csgraph.connected_components(treshold_matrix, directed=False)

            fire_cluster_ids = fire_clusters + n_fires_total # Offset cluster ids by total ids already assigned
            data.loc[df_year.index, 'cluster_id'] = fire_cluster_ids

            n_fires_total += n_fires_year
            logging.debug('Found %d unique clusters for year %d' % (n_fires_year, year))

        logging.debug('Found %d unique clusters for all years' % n_fires_total)
        return data

    def build_cluster_df(self, df):
        logging.debug('Starting building of final cluster data frame')

        cluster_df_data = []

        cluster_ids = range(int(np.max(df.cluster_id))+1)
        for cluster_id in cluster_ids:
            logging.debug('Starting processing of cluster %d/%d' % (cluster_id+1, len(cluster_ids)))
            c_df = df[df.cluster_id==cluster_id]
            lat_centroid, lon_centroid = np.mean(c_df.lat), np.mean(c_df.lon)

            #dates = set(c_df.date_local)
            dates = du.daterange(np.min(c_df.date_local), np.max(c_df.date_local))
            for date in dates:
                date_df = c_df[c_df.date_local==date]
                if not date_df.empty:
                    num_det = len(date_df)
                    avg_frp = np.mean(date_df.FRP)
                    avg_conf = np.mean(date_df.conf)
                else:
                    num_det, avg_frp, avg_conf = 0, np.nan, np.nan

                cluster_df_data.append((date, cluster_id, num_det, lat_centroid, lon_centroid, avg_frp, avg_conf))

        cluster_df = pd.DataFrame(cluster_df_data, columns=['date_local', 'cluster_id', 'num_det', 'lat_centroid', 'lon_centroid', 'avg_frp', 'avg_conf'])

        return cluster_df


@click.command()
@click.argument('src_path', type=click.Path(exists=True))
@click.argument('dest_path')
@click.option('--cluster', default=None)
@click.option('--log', default='INFO')
def main(src_path, dest_path, cluster, log):
    """
    Load fire data frame and create clusters.
    """
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging, log.upper()), format=log_fmt)

    logging.info('Starting fire data frame to cluster conversion')
    FireDfToClusterConverter(cluster).convert(src_path, dest_path)
    logging.info('Finished fire data frame to cluster conversion')


if __name__ == '__main__':
    main()
