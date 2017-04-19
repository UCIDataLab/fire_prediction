import numpy as np

# constants
ak_bb = [55, 71, -165, -138]
km_per_deg_lat = 111


def km_per_deg_lon(lat): np.cos(lat*np.pi/180.) * 111.321


def get_grid_fxns(bb, grid_res=1.1):
    """ Get conversion functions from lat/lon to a uniform distance grid. Note that because longitude
    shrinks as we move up in latitude, the northeast grid cells will be invalid
    :param bb: bounding box of area to grid-ify
    :param grid_res: resolution of the grid (in km)
    :return: latlon2grid, grid2latlon, grid_shape
    """
    y_shape = (bb[1] - bb[0]) * km_per_deg_lat / grid_res
    x_shape = (bb[3] - bb[2]) * km_per_deg_lon(bb[0]) / grid_res

    def latlon2grid(lat, lon):
        y = int((lat - bb[0]) * km_per_deg_lat / grid_res)
        km_per_lon_here = km_per_deg_lon(lat)
