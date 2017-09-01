"""
Code for handling global weather data.
"""
import bisect

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

        new = WeatherCube(self.name, self.values[:,:,ind_start:ind_end], self.units, self.bounding_box, self.axis_labels, self.dates[ind_start:ind_end])
        for k,v in self.attributes.iteritems():
            if len(v) == len(self.dates):
                new.add_attribute(k, v[ind_start:ind_end])
            else:
                new.add_attribute(k, v)

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


class WeatherRegion(object):
    """
    A collection of WeatherCubes with matching shape and dates.
    """

    def __init__(self, name, cubes={}):
        self.name = name

        self.cubes = {}
        for _,cube in cubes.iteritems():
            self.add_cube(cube)
            
        self.shape = None
        self.dates = None
        self.bounding_box = None

    # TODO: Free memory for repeated shape, dates and bounding_box?
    def add_cube(self, cube):
        if self.shape and (cube.shape != self.shape):
            raise ValueError('All cubes in a region must have the same shape. Cube shape %s != region shape %s' % (cube.shape, self.shape))

        if self.dates and (cube.dates != self.dates):
            raise ValueError('All cubes in a region must have the same dates' % (np.shape(cube.data), self.shape))

        if self.bounding_box and (cube.bounding_box != self.bounding_box):
            raise ValueError('All cubes in a region must have the same bounding box. Cube bb %s != region bb %s' % (str(cube.bounding_box), str(self.bounding_box)))

        self.shape = cube.shape
        self.dates = cube.dates
        self.bounding_box = cube.bounding_box

        self.cubes[cube.name] = cube

    def remove_cube(self, cube_name):
        del self.cubes[cube_name]

    def __getitem__(self, key):
        return self.cubes[key]

    def filter_dates(self, date_start, date_end):
        ind_start = bisect.bisect_left(self.dates, date_start)
        ind_end = bisect.bisect_right(self.dates, date_end)

        new_cubes = {}
        for name,cube in self.cubes.iteritems():
            new_cubes[name] = cube.filter_dates(date_start, date_end)

        new = WeatherRegion(self.name, new_cubes)

        return new


