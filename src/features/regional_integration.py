import click
import logging
import pandas as pd
import datetime as dt
from datetime import datetime
import numpy as np
import cPickle as pickle

import pytz

import helper.date_util as du
class FireWeatherIntegrationRegional(object):
    def __init__(self, k_days, time, fill_missing, fill_n_days, rain_offset, weather_vars_labels=['temperature', 'humidity', 'wind', 'rain']):
        self.k_days = k_days
        self.time = time
        self.fill_missing = fill_missing
        self.fill_n_days = fill_n_days
        self.rain_offset = rain_offset
        self.weather_vars_labels = weather_vars_labels

        self.integrated_data = None

    def integrate(self, fire_src_path, weather_src_path):


@click.command()
@click.argument('fire_src_path', type=click.Path(exists=True))
@click.argument('weather_src_path', type=click.Path(exists=True))
@click.argument('dest_path')
@click.option('--k', default=1, type=click.INT)
@click.option('--time', default=14, type=click.INT)
@click.option('--fill', default=True, type=click.BOOL)
@click.option('--filldays', default=5, type=click.INT)
@click.option('--rainoffset', default=0, type=click.INT)
@click.option('--log', default='INFO')
def main(fire_src_path, weather_src_path, dest_path, k, time, mean, fill, filldays, rainoffset, log):
    """
    Load fire data frame and create clusters.
    """
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging, log.upper()), format=log_fmt)

    fwi = FireWeatherIntegrationRegional(k, time, fill, filldays, rainoffset)

    logging.info('Starting fire/weather integration regional for k=%d' % k)
    fwi.integrate(fire_src_path, weather_src_path)
    fwi.save(dest_path)
    logging.info('Finished fire/weather integration regional')


if __name__ == '__main__':
    main()
