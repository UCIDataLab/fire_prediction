import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def draw_map_nogrid_static(bb, month=1, day=1, gfs_dict=None, points=None, marker=None, draw_terrain=False):
    """ Plot a map with optional points and optional overlays
    :param bb: bounding box of region to plot of the form [min_lat, max_lat, min_lon, max_lon]
    :param month: Month we will plot
    :param day: Day we will plot
    :param gfs_dict: A dictionary from (month,day) tuples to gfs overlay values. Also contains special entries 'lat',
        'lon', 'max', and 'min'
    :param points: Tuple of lat and lon coordinates for points to plot. If None, plot no points
    :param marker: Marker string to use for points
    :param draw_terrain: Boolean for whether or not to plot the terrain
    :return:
    """
    if gfs_dict:
        lats = gfs_dict['lats']
        lons = gfs_dict['lons']
        n_lat, n_lon = lats.shape
        gfs_min = gfs_dict['min']
        gfs_max = gfs_dict['max']
        plot_bb_0 = np.where(lats[:,0] <= bb[0])[0][0]
        plot_bb_1 = np.where(lats[:,0] <= bb[1])[0][0]
        plot_bb_2 = np.where(lons[0,:] >= (bb[2] % (n_lon/2)))[0][0]
        plot_bb_3 = np.where(lons[0,:] >= (bb[3] % (n_lon/2)))[0][0]
        plot_bb = [plot_bb_0, plot_bb_1, plot_bb_2, plot_bb_3]
    mp = Basemap(projection="merc",
                  lat_0=55, lon_0=-165,
                  llcrnrlat=55,
                  llcrnrlon=-165,
                  urcrnrlat=71,
                  urcrnrlon=-138,
                  resolution='i')
    mp.drawcoastlines()
    mp.imshow(temp_vals[plot_bb_0-1:plot_bb_1-1:-1, plot_bb_2:plot_bb_3])
    mp.colorbar()