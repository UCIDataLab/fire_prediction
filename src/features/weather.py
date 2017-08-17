"""
Code for handling global weather data.
"""

class WeatherCube(object):
    def __init__(self, name, values, units, bounding_box=None, axis_labels=None, dates=None):
        self.name = name
        self.values = values
        self.units = units
        self.bounding_box = bounding_box
        self.axis_labels = axis_labels
        self.dates = dates

class WeatherRegion(object):
    def __init__(self, name):
        self.name = name
        self.cubes = {}
        self.shape = None

    def add_cube(self, cube):
        if self.shape:
            if np.shape(cube.data) != self.shape:
                raise ValueError('All cubes in a region must have the same shape. Cube shape %s != region shape' % (np.shape(cube.data), self.shape))

        self.cubes[cube.name] = cube

    def __getitem__(self, key):
        return self.cubes[key]
