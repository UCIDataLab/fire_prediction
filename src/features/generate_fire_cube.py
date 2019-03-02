"""
Generates a dense grid of MODIS detections (with time axis).
"""

import datetime as dt
import logging

import helper.date_util as du
import helper.df_util as dfu
import numpy as np
import pandas as pd
import xarray as xr


class GridGenerator(object):
    def __init__(self, bounding_box):
        super(GridGenerator, self).__init__()

        self.bounding_box = bounding_box

    def transform(self, data):
        logging.debug('Applying transforms to data')

        # Add local datetime col to determine the day the fire occured in
        df = data.assign(date_local=list(
            map(lambda x: du.utc_to_local_time(x[0], x[1], du.round_to_nearest_quarter_hour).date(),
                zip(data.datetime_utc, data.lon))))

        # Create grid
        # TODO: add dynamic sizing (bounds and increment)
        lats, lons = self.bounding_box.make_grid(inclusive_lon=True)
        spatial_size = np.shape(lats)

        year_range = dfu.get_year_range(df, 'datetime_utc')
        dates = list(du.daterange(dt.date(year_range[0], 1, 1), dt.date(year_range[1] + 1, 1, 1)))
        grid = np.zeros(spatial_size + (len(dates),))

        # Assign each detection to a cell in time and space
        for row in df.itertuples():
            lat, lon, date = row.lat, row.lon, row.date_local

            lat_ind, lon_ind = self.bounding_box.latlon_to_indices(lat, lon,
                                                                   spatial_size[0])  # TODO: add dynamic sizing
            date_ind = (date - dates[0]).days

            grid[lat_ind, lon_ind, date_ind] += 1

        ds = xr.Dataset({'detections': (['y', 'x', 'time'], grid)},
                        coords={'y': lats[:, 0], 'x': lons[0, :], 'time': pd.to_datetime(dates)})

        ds.attrs['bounding_box'] = self.bounding_box.get()

        return ds
