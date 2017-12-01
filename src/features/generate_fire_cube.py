"""
Generates a dense grid of MODIS detections (with time axis).
"""

import numpy as np
import click
import logging
import cPickle as pickle
import datetime as dt

import helper.df_util as dfu
import helper.geometry as geo
import helper.date_util as du
from helper import weather

import bisect

from base.converter import Converter

class GridGenerator(Converter):
    def __init__(self, ignitions_only):
        super(GridGenerator, self).__init__()

        self.ignitions_only = ignitions_only

    def load(self, src_path):
        logging.info('Loading file from %s' % src_path)

        with open(src_path, 'rb') as fin:
            data = pickle.load(fin)

        return data
    
    def save(self, dest_path, data):
        logging.info('Saving data to %s' % dest_path)

        with open(dest_path, 'wb') as fout:
            pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)

    def transform(self, data):
        logging.debug('Applying transforms to data')

        # Add local datetime col to determine the day the fire occured in
        df = data.assign(date_local=map(lambda x: du.utc_to_local_time(x[0], x[1], du.round_to_nearest_quarter_hour).date(), zip(data.datetime_utc, data.lon)))

        # Create grid
        year_range = dfu.get_year_range(df, 'datetime_utc')
         # TODO: add dynamic sizing (bounds and increment)
        bb = geo.LatLonBoundingBox(55, 71, -165, -138)
        spatial_size = np.shape(bb.make_grid()[0])
        dates = list(du.daterange(dt.date(year_range[0], 1, 1), dt.date(year_range[1]+1, 1, 1)))
        grid = np.zeros(spatial_size + (len(dates),))

        if self.ignitions_only:
            num_clusters = int(np.max(df['cluster_id']))
            clusters_seen = np.zeros(num_clusters)

        # Assign each detection to a cell in time and space
        for row in df.itertuples():
            lat,lon,date = row.lat, row.lon, row.date_local
            if self.ignitions_only:
                if clusters_seen[int(row.cluster_id)-1]:
                    continue
                else:
                    clusters_seen[int(row.cluster_id)-1] = 1

            lat_ind,lon_ind = bb.latlon_to_indices(lat,lon,spatial_size[0]) # TODO: add dynamic sizing
            date_ind = (date-dates[0]).days

            grid[lat_ind,lon_ind,date_ind] += 1

        return weather.WeatherCube('detections', grid, 'det', dates=dates)


def gen_grid_predictions(preds, info, fire_season=((5,14), (8,31))):
    # Create grid
    year = info[0][0].year

     # TODO: add dynamic sizing (bounds and increment)
    bb = geo.LatLonBoundingBox(55, 71, -165, -138)
    spatial_size = np.shape(bb.make_grid()[0])
    dates = list(du.daterange(dt.date(year, fire_season[0][0], fire_season[0][1]),
        dt.date(year, fire_season[1][0], fire_season[1][1]) + du.INC_ONE_DAY))
    grid = np.zeros(spatial_size + (len(dates),))

    # Assign each detection to a cell in time and space
    for pred, row in zip(preds,info):
        lat,lon,date = row[1], row[2], row[0] 
        lat_ind,lon_ind = bb.latlon_to_indices(lat,lon,spatial_size[0]) # TODO: add dynamic sizing
        date_ind = (date-dates[0]).days

        grid[lat_ind,lon_ind,date_ind] += pred

    return grid, dates
        

@click.command()
@click.argument('src_path', type=click.Path(exists=True))
@click.argument('dest_path')
@click.option('--ignitions_only', default=False, type=click.BOOL)
@click.option('--log', default='INFO')
def main(src_path, dest_path, ignitions_only, log):
    """
    Load aggregated fire data and generate grid.
    """
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging, log.upper()), format=log_fmt)

    logging.info('Starting fire data frame to grid conversion. Ignitions Only: %r' % ignitions_only)
    GridGenerator(ignitions_only).convert(src_path, dest_path)
    logging.info('Finished fire data frame to grid conversion')


if __name__ == '__main__':
    main()
