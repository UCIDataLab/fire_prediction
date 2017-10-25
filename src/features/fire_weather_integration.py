"""
Integrating clustered fire data and weather data.
"""

import click
import logging
import pandas as pd
import datetime as dt
from datetime import datetime
import numpy as np
import cPickle as pickle

import pytz

import helper.date_util as du

class FireWeatherIntegration(object):
    def __init__(self, time, fill_missing, fill_n_days, weather_vars_labels=['temperature', 'humidity', 'wind', 'rain']):
        self.time = time
        self.fill_missing = fill_missing
        self.fill_n_days = fill_n_days
        self.weather_vars_labels = weather_vars_labels

        self.integrated_data = None

    def integrate(self, fire_src_path, weather_src_path):
        logging.info('Integerating fire and weather data')
        fire_df = self.load_fire(fire_src_path)

        # TEMP: Reset index numbering for fire_df
        fire_df.reset_index(drop=True, inplace=True)

        weather_region = self.load_weather(weather_src_path)

        weather_vars = []
        for i, row in enumerate(fire_df.itertuples()):
            logging.debug('Starting integration for row %d/%d' % (i+1, fire_df.shape[0]))
            date, lat, lon = row.date_local, row.lat_centroid, row.lon_centroid

            target_datetime = datetime.combine(date, dt.time(self.time, 0, 0, tzinfo=du.TrulyLocalTzInfo(lon, du.round_to_nearest_quarter_hour)))

            var = self.get_weather_variables(weather_region, target_datetime, lat, lon)
            weather_vars.append(var)
        
        weather_df = pd.DataFrame(weather_vars, columns=self.weather_vars_labels)

        self.integrated_data = fire_df.join(weather_df, how='outer')

    def load_fire(self, src_path):
        logging.info('Loading file from %s' % src_path)
        with open(src_path, 'rb') as fin:
            data = pickle.load(fin)

        return data

    def load_weather(self, src_path):
        logging.info('Loading file from %s' % src_path)
        with open(src_path, 'rb') as fin:
            data = pickle.load(fin)

        return data

    def save(self, dest_path):
        logging.info('Saving data frame to %s' % dest_path)

        with open(dest_path, 'wb') as fout:
            pickle.dump(self.integrated_data, fout, protocol=pickle.HIGHEST_PROTOCOL)

    def get_weather_variables(self, weather_data, target_datetime, lat, lon):
        # Get lat/lon index
        lat_ind, lon_ind = self.get_latlon_index(weather_data, lat, lon)

        # Get date index
        date_ind = self.get_date_index(weather_data, target_datetime)

        vals = []
        for key in self.weather_vars_labels:
            data = weather_data[key].values
            val = data[lat_ind, lon_ind, date_ind]

            if np.isnan(val) and self.fill_missing:
                val = self.fill_missing_value(data, lat_ind, lon_ind, date_ind)

            vals.append(val)

        return vals

    def fill_missing_value(self, data, lat_ind, lon_ind, date_ind):
        """
        Try to replace with closest prev day in range [1, fill_n_days].

        If no non-nan value is found, replaces with mean of all values at the given lat/lon.
        """
        for day_offset in range(1,self.fill_n_days+1):
            new_date_ind = date_ind - day_offset

            if new_date_ind < 0:
                break

            val = data[lat_ind, lon_ind, new_date_ind]

            if not np.isnan(val):
                return val

        return np.nanmean(data[lat_ind, lon_ind, :])

    def get_latlon_index(self, weather_data, lat, lon):
        bb = weather_data.bounding_box


        lat_res, lon_res = bb.get_latlon_resolution(weather_data.shape[:2])
        lat_min, lat_max, lon_min, lon_max = bb.get()

        if (lat > lat_max) or (lat < lat_min) or (lon > lon_max) or (lon < lon_min):
            raise ValueError('Lat or lon outside of bounding box.')

        lat_ind = int(round(float(abs(lat_max - lat)) / lat_res))
        lon_ind = int(round(float(abs(lon_min - lon)) / lon_res))

        return lat_ind, lon_ind

    def get_date_index(self, weather_data, target_datetime):
        date_ind = np.searchsorted(weather_data.dates, target_datetime, side='left')

        # Check if left or right element is closer
        if date_ind != 0:
            date_ind_left, date_ind_curr = date_ind-1, date_ind

            dist_left = abs((weather_data.dates[date_ind_left] - target_datetime).total_seconds())
            dist_curr = abs((weather_data.dates[date_ind_curr] - target_datetime).total_seconds())

            if dist_left < dist_curr:
                date_ind = date_ind_left

        return date_ind


@click.command()
@click.argument('fire_src_path', type=click.Path(exists=True))
@click.argument('weather_src_path', type=click.Path(exists=True))
@click.argument('dest_path')
@click.option('--time', default=14, type=click.INT)
@click.option('--fill', default=True, type=click.BOOL)
@click.option('--filldays', default=5, type=click.INT)
@click.option('--log', default='INFO')
def main(fire_src_path, weather_src_path, dest_path, time, fill, filldays, log):
    """
    Load fire data frame and create clusters.
    """
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging, log.upper()), format=log_fmt)

    fwi = FireWeatherIntegration(time, fill, filldays)

    logging.info('Starting fire/weather integration')
    fwi.integrate(fire_src_path, weather_src_path)
    fwi.save(dest_path)
    logging.info('Finished fire/weather integration')


if __name__ == '__main__':
    main()
