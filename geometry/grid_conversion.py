import numpy as np
from util.daymonth import day2monthday

# constants
ak_bb = [55, 71, -165, -138]
west_coast_bb = [32,50,-125,-110]
km_per_deg_lat = 111.


def km_per_deg_lon(lat): return np.cos(lat*np.pi/180.) * 111.321


def get_latlon_xy_fxns(bb):
    """ Get conversion functions from lat/lon to a uniform distance grid. Note that because longitude
    shrinks as we move up in latitude, the northeast/northwest grid cells will be invalid
    :param bb: bounding box of area to grid-ify
    :return: latlon2xy, xy2latlon, (X,Y)
    """
    y_shape = (bb[1] - bb[0]) * km_per_deg_lat
    x_shape = (bb[3] - bb[2]) * km_per_deg_lon(bb[0])
    center_lon = float(bb[2] + bb[3]) / 2
    center_x = x_shape / 2

    def latlon2xy(lat, lon):
        y = (lat - bb[0]) * km_per_deg_lat
        km_per_lon_here = km_per_deg_lon(lat)
        x = center_x - ((center_lon - lon) * km_per_lon_here)
        return x,y

    def xy2latlon(x,y):
        lat = bb[0] + (y / km_per_deg_lat)
        km_per_lon_here = km_per_deg_lon(lat)
        lon = center_lon - (float(center_x - x) / km_per_lon_here)
        return lat,lon

    return latlon2xy, xy2latlon, (x_shape, y_shape)


def get_latlon_grid_fxns(bb, grid_res=1.1):
    """ Get conversion functions from lat/lon to a uniform distance grid. Note that because longitude
    shrinks as we move up in latitude, the northeast/northwest grid cells will be invalid
    :param bb: bounding box of area to grid-ify
    :param grid_res: resolution of grid to create
    :return: latlon2xy, xy2latlon, (X,Y)
    """
    y_shape = int((bb[1] - bb[0]) * km_per_deg_lat / grid_res)
    x_shape = int((bb[3] - bb[2]) * km_per_deg_lon(bb[0]) / grid_res)
    center_lon = float(bb[2] + bb[3]) / 2
    center_x = x_shape / 2  #int(center_lon * km_per_deg_lon(bb[0]) / grid_res)

    def latlon2grid(lat, lon):
        y = int((lat - bb[0]) * km_per_deg_lat / grid_res)
        km_per_lon_here = km_per_deg_lon(lat)
        x = int(center_x - (((center_lon - lon) * km_per_lon_here) / grid_res))
        return x,y

    def grid2latlon(x,y):
        lat = bb[0] + (y * grid_res / km_per_deg_lat)
        km_per_lon_here = km_per_deg_lon(lat)
        lon = center_lon - (float(center_x - x) * grid_res / km_per_lon_here)
        return lat,lon

    return latlon2grid, grid2latlon, (x_shape, y_shape)


def get_gfs_val(lat, lon, day, month, gfs_dict, year=2013):
    """ Find the GFS value for the lat/lon nearest to the one specified
    :param lat: latitude
    :param lon: longitude
    :param day: day
    :param gfs_dict: dict of gfs vals
    :return: val from gfs
    """
    if (month,day,year) not in gfs_dict:
        raise KeyError("%d/%d/%d not in gfs dict" % (month,day,year))
    lats = gfs_dict['lats']
    lons = gfs_dict['lons']
    n_lat, n_lon = lats.shape
    lat_res = lats[0,0] - lats[1,0]
    lon_res = lons[0,1] - lons[0,0]
#    print "lat %s lon %s lat_res %s lon_res %s" % (str(lat),str(lon),str(lat_res),str(lon_res))
    row = int(float(lats[0,0] - lat) / lat_res)
    positive_lon = lon % 360   # convert longitude to a positive value, which is what GFS uses
    col = int(float(lons[0,0] - positive_lon) / lon_res)
    return gfs_dict[(month,day,year)][row,col]
