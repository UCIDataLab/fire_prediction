"""
Code for handling global weather data.
"""
import bisect
import datetime as dt

import numpy as np
from helper import date_util as du


class WeatherCube(object):
    def __init__(self, name, values, units, bounding_box=None, axis_labels=None, dates=None):
        """
        Assumes that dates is as long as values and is ordered.
        """
        self.name = name
        self.values = values
        self.units = units
        self.bounding_box = bounding_box
        self.axis_labels = axis_labels
        self.dates = dates

        self.attributes = {}
        self.shape = values.shape

    def add_attribute(self, name, value):
        self.attributes[name] = value

    def filter_dates(self, date_start, date_end):
        ind_start = bisect.bisect_left(self.dates, date_start)
        ind_end = bisect.bisect_right(self.dates, date_end)

        new = WeatherCube(self.name, self.values[:, :, ind_start:ind_end], self.units, self.bounding_box,
                          self.axis_labels, self.dates[ind_start:ind_end])
        for k, v in self.attributes.items():
            if len(v) == len(self.dates):
                new.add_attribute(k, v[ind_start:ind_end])
            else:
                new.add_attribute(k, v)

        return new

    def filter_dates_per_year(self, date_range):
        # TODO: No support for attributes.
        years = range(np.min(self.dates).year, np.max(self.dates).year + 1)
        values = []
        dates = []
        for year in years:
            date_start, date_end = dt.date(year, date_range[0][0], date_range[0][1]), dt.date(year, date_range[1][0],
                                                                                              date_range[1][1])
            ind_start = bisect.bisect_left(self.dates, date_start)
            ind_end = bisect.bisect_right(self.dates, date_end)

            values.append(self.values[:, :, ind_start:ind_end])
            dates.append(self.dates[ind_start:ind_end])

        new = WeatherCube(self.name, np.concatenate(values, axis=2), self.units, self.bounding_box, self.axis_labels,
                          np.concatenate(dates, axis=0))

        return new

    def get_values_for_date(self, date):
        """
        Return all values with matching date.
        """
        ind_start = bisect.bisect_left(self.dates, date)
        ind_end = bisect.bisect_right(self.dates, date)

        return self.values[:, :, ind_start:ind_end]

    def get_attribute_for_date(self, name, date):
        """
        Return all values with matching date.
        """
        # TODO support per item attributes
        attribute = self.attributes[name]

        ind_start = bisect.bisect_left(self.dates, date)
        ind_end = bisect.bisect_right(self.dates, date)

        return attribute[ind_start:ind_end]

    def remove_year(self, year):
        """
        :param year: Year to remove from cube
        
        :return: (new WeatherCube w/o year, new WeatherCube w/ only year)
        """
        year_start = dt.date(year, 1, 1)
        year_end = dt.date(year + 1, 1, 1) - du.INC_ONE_DAY

        ind_start = bisect.bisect_left(self.dates, year_start)
        ind_end = bisect.bisect_right(self.dates, year_end)

        # Build WeatherCube with only year included
        year_only = WeatherCube(self.name, self.values[:, :, ind_start:ind_end], self.units, self.bounding_box,
                                self.axis_labels, self.dates[ind_start:ind_end])
        for k, v in self.attributes.items():
            if len(v) == len(self.dates):
                year_only.add_attribute(k, v[ind_start:ind_end])
            else:
                year_only.add_attribute(k, v)

        # Build WeatherCube w/o year included
        vals = np.concatenate((self.values[:, :, :ind_start], self.values[:, :, ind_end:]), axis=2)
        dates = np.concatenate((self.dates[:ind_start], self.dates[ind_end:]), axis=0)
        year_not = WeatherCube(self.name, vals, self.units, self.bounding_box, self.axis_labels, dates)
        for k, v in self.attributes.items():
            if len(v) == len(self.dates):
                v_ = np.concatenate((v[:ind_start], v[ind_end:]), axis=0)
                year_not.add_attribute(k, v_)
            else:
                year_not.add_attribute(k, v)

        return year_not, year_only


class WeatherRegion(object):
    """
    A collection of WeatherCubes with matching shape and dates.
    """

    def __init__(self, name, cubes=None):
        if cubes is None:
            cubes = {}

        self.name = name

        self.shape = None
        self.dates = None
        self.bounding_box = None

        # TODO: Remove this, hack for cross-validation supporting Cubes or Regions
        self.values = self

        self.cubes = {}
        for _, cube in cubes.items():
            self.add_cube(cube)

    # TODO: Free memory for repeated shape, dates and bounding_box?
    def add_cube(self, cube):
        if self.shape is not None and np.any(cube.shape != self.shape):
            raise ValueError('All cubes in a region must have the same shape. Cube shape %s != region shape %s' % (
                cube.shape, self.shape))

        if self.dates is not None and np.any(cube.dates != self.dates):
            raise ValueError('All cubes in a region must have the same dates')

        if self.bounding_box is not None and np.any(cube.bounding_box != self.bounding_box):
            raise ValueError('All cubes in a region must have the same bounding box. Cube bb %s != region bb %s' % (
                str(cube.bounding_box), str(self.bounding_box)))

        self.shape = cube.shape
        self.dates = cube.dates
        self.bounding_box = cube.bounding_box

        self.cubes[cube.name] = cube

    def remove_cube(self, cube_name):
        del self.cubes[cube_name]

    def __getitem__(self, key):
        return self.cubes[key]

    def filter_dates(self, date_start, date_end):
        new_cubes = {}
        for name, cube in self.cubes.items():
            new_cubes[name] = cube.filter_dates(date_start, date_end)

        new = WeatherRegion(self.name, new_cubes)

        return new

    def remove_year(self, year):
        """
        :param year: Year to remove from cube

        :return: (new WeatherCube w/o year, new WeatherCube w/ only year)
        """
        new_cubes_without_year = {}
        new_cubes_with_year = {}
        for name, cube in self.cubes.items():
            cube_without_year, cube_with_year = cube.remove_year(year)
            new_cubes_without_year[name] = cube_without_year
            new_cubes_with_year[name] = cube_with_year

        new_without_year = WeatherRegion(self.name, new_cubes_without_year)
        new_with_year = WeatherRegion(self.name, new_cubes_with_year)

        return new_without_year, new_with_year
