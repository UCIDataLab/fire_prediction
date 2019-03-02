"""
Processing the weather data for integration.
"""
import logging

import pickle
import click
import numpy as np
from .base.converter import Converter
from ..helper import weather


class WeatherRegionProcessing(Converter):
    """
    Process a weather region to prepare it for integration.
    """

    def __init__(self):
        super(WeatherRegionProcessing, self).__init__()

    @staticmethod
    def load(src_path):
        logging.info('Loading file from %s' % src_path)
        with open(src_path, 'rb') as fin:
            data = pickle.load(fin)

        return data

    @staticmethod
    def save(dest_path, data):
        logging.info('Saving data frame to %s' % dest_path)

        with open(dest_path, 'wb') as fout:
            pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)

    def transform(self, data):
        logging.debug('Applying transforms to data frame')
        self.combine_wind_components(data)
        self.integrate_rain(data)
        data = self.remove_offset_measurements(data)

        return data

    @staticmethod
    def combine_wind_components(data):
        logging.debug('Combining wind component')
        u_wind_cube = data['u_wind_component']
        v_wind_cube = data['v_wind_component']

        wind_values = np.sqrt(u_wind_cube.values ** 2 + v_wind_cube.values ** 2)
        wind_cube = weather.WeatherCube('wind', wind_values, u_wind_cube.units, u_wind_cube.bounding_box,
                                        u_wind_cube.axis_labels, u_wind_cube.dates)

        data.remove_cube('u_wind_component')
        data.remove_cube('v_wind_component')

        data.add_cube(wind_cube)

    @staticmethod
    def integrate_rain(data):
        logging.debug('Integrating rain')
        rain_cube = data['total_precipitation']

        integrated_rain_values = np.empty(rain_cube.shape)
        integrated_rain_values.fill(np.nan)

        for i in range(12, rain_cube.shape[2], 3):
            val = rain_cube.values[:, :, i - 10] + rain_cube.values[:, :, i - 7] +\
                  rain_cube.values[:, :, i - 4] + rain_cube.values[:, :, i - 1]
            integrated_rain_values[:, :, i] = val

        integrated_rain_cube = weather.WeatherCube('rain', integrated_rain_values, rain_cube.units,
                                                   rain_cube.bounding_box, rain_cube.axis_labels, rain_cube.dates)

        data.remove_cube('total_precipitation')

        data.add_cube(integrated_rain_cube)

    def remove_offset_measurements(self, data):
        """
        Remove all entries for non-zero offsets. Also replaces DatetimeMeasurements with just the datetime componenet.
        """
        logging.debug('Removing offset measurements')
        new_region = weather.WeatherRegion(data.name)

        for _, cube in data.cubes.items():
            new_cube = self.remove_offset_measurements_cube(cube)
            new_region.add_cube(new_cube)

        return new_region

    @staticmethod
    def remove_offset_measurements_cube(cube):
        """
        Remove all entries for non-zero offsets from a cube. Also replaces DatetimeMeasurements with just the datetime
        component.
        """
        if cube.shape[2] % 3 != 0:
            raise ValueError(
                'Cube shape\'s third dimension must be divisible by three (+0, +3, +6 offsets). '
                'Shape of {} is {}.'.format(cube.name, str(cube.shape)))

        new_values = cube.values[:, :, ::3].copy()
        new_dates = map(lambda x: x.get(), cube.dates[::3])

        new_cube = weather.WeatherCube(cube.name, new_values, cube.units, cube.bounding_box, cube.axis_labels,
                                       new_dates)

        return new_cube


@click.command()
@click.argument('src_path', type=click.Path(exists=True))
@click.argument('dest_path')
@click.option('--log', default='INFO')
def main(src_path, dest_path, log):
    """
    Load fire data frame and create clusters.
    """
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging, log.upper()), format=log_fmt)

    logging.info('Starting weather processing')
    WeatherRegionProcessing().convert(src_path, dest_path)
    logging.info('Finished weather processing')


if __name__ == '__main__':
    main()
