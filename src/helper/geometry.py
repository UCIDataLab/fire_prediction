"""
Handling spatial measurements and calculations. Especially related to maps and lon/lat.
"""
import numpy as np


class LatLonBoundingBox(object):
    """
    Stores lat/lon bounding-box info.

    Lat in [-90,90], Lon in [-180, 180].
    """

    def __init__(self, lat_min=-90.0, lat_max=90.0, lon_min=-180.0, lon_max=180.0):
        self.lat_min = float(lat_min)
        self.lat_max = float(lat_max)
        self.lon_min = float(lon_min)
        self.lon_max = float(lon_max)

        if lat_max < lat_min:
            raise ValueError('lat_max (%f) less than lat_min (%f)' % (lat_max, lat_min))
        if lon_max < lon_min:
            raise ValueError('lon_max (%f) less than lon_min (%f)' % (lon_max, lon_min))

    def get(self):
        return self.lat_min, self.lat_max, self.lon_min, self.lon_max

    def get_min_max_indexes(self, diff_lat, diff_lon):
        """
        Get indices for diff_lat and diff_lon where min/max lon/lat can be found.
        """
        lat_min_ind = self.get_index(diff_lat, self.lat_min, np.argmin)
        lat_max_ind = self.get_index(diff_lat, self.lat_max, np.argmax)
        lon_min_ind = self.get_index(diff_lon, self.lon_min, np.argmin)
        lon_max_ind = self.get_index(diff_lon, self.lon_max, np.argmax)

        return lat_min_ind, lat_max_ind, lon_min_ind, lon_max_ind

    @staticmethod
    def get_index(distinct, val, default_func):
        try:
            ind = np.where(distinct == val)[0][0]
        except IndexError:
            ind = default_func(distinct)

        return int(ind)

    def latlon_to_indices(self, lat, lon, num_lat, lat_inc=.5, lon_inc=.5):
        if lat > self.lat_max or lat < self.lat_min:
            raise ValueError('Lat out of range: %f' % lat)
        elif lon > self.lon_max or lon < self.lon_min:
            raise ValueError('Lon out of range: %f' % lon)

        lat_ind = (num_lat - 1) - round((lat - self.lat_min) / lat_inc)
        lon_ind = round((lon - self.lon_min) / lon_inc)

        return int(lat_ind), int(lon_ind)

    def make_grid(self, lat_inc=.5, lon_inc=.5, inclusive_lon=False):
        # TODO: check for divisibility by increments
        lats = np.arange(self.lat_max, self.lat_min - lat_inc, -lat_inc)

        lon_upper_bound = self.lon_max + lon_inc if inclusive_lon else self.lon_max
        lons = np.arange(self.lon_min, lon_upper_bound, lon_inc)

        num_lats, num_lons = len(lats), len(lons)

        lats = np.transpose(np.tile(lats, (num_lons, 1)))
        lats = np.array(lats, order='c')  # Basemap has issues with Fortran order created by transposing
        lons = np.tile(lons, (num_lats, 1))

        return lats, lons

    def get_latlon_resolution(self, latlon_shape):
        """
        latlon_shape should contain two elements, the length of the lat dim and the length of the lon dim
        """
        lat_min, lat_max, lon_min, lon_max = self.get()
        lat_range, lon_range = lat_max - lat_min, lon_max - lon_min
        lat_res, lon_res = lat_range / float(latlon_shape[0] - 1), lon_range / float(latlon_shape[1] - 1)

        return lat_res, lon_res

    def __str__(self):
        return str({'lat': (self.lat_min, self.lat_max), 'lon': (self.lon_min, self.lon_max)})

    def __eq__(self, other):
        if isinstance(other, LatLonBoundingBox):
            return self.get() == other.get()
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.get())


def latlon_range(bounding_box, inc_lat=1., inc_lon=1.):
    lat_min, lat_max, lon_min, lon_max = bounding_box.get()

    lat_min -= inc_lat
    lon_max += inc_lon

    lon_min_orig = lon_min

    while lat_min < lat_max:
        while lon_min < lon_max:
            yield lat_max, lon_min
            lon_min += inc_lon

        lat_max -= inc_lat
        lon_min = lon_min_orig


def get_default_bounding_box():
    return LatLonBoundingBox(55, 71, -165, -138)


def filter_bounding_box_df(df, bb):
    """
    Filter data frame with 'lat' and 'lon' columns to only include values inside bounding box (inclusive).
    """
    min_lat, max_lat, min_lon, max_lon = bb.get()
    return df[(df.lat <= max_lat) & (df.lat >= min_lat) & (df.lon <= max_lon) & (df.lon >= min_lon)]


def upsample_spatial(target_shape, data):
    target_shape = (data.shape[0],) + target_shape
    upsampled = np.empty(target_shape, dtype=data.dtype)

    # TODO: Look into this upsampling more
    x_ratio = np.ceil(target_shape[1] / data.shape[1]).astype(np.int32)
    y_ratio = np.ceil(target_shape[2] / data.shape[2]).astype(np.int32)

    for (x, y) in [(x, y) for x in range(target_shape[1]) for y in range(target_shape[2])]:
        upsampled[:, x, y] = data[:, x // x_ratio, y // y_ratio]

    return upsampled
