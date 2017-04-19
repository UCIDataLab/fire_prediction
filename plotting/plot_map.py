import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.basemap import Basemap
from util.daymonth import day2monthday, monthday2day
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
    mp = Basemap(projection="merc",
                 lat_0=bb[0], lon_0=bb[2],
                 llcrnrlat=bb[0],
                 llcrnrlon=bb[2],
                 urcrnrlat=bb[1],
                 urcrnrlon=bb[3],
                 resolution='i')
    mp.drawcoastlines()
    if draw_terrain:
        mp.etopo()
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
        vals = gfs_dict[(month, day)]
        mp.imshow(vals[plot_bb_0-1:plot_bb_1-1:-1, plot_bb_2:plot_bb_3], vmin=gfs_min, vmax=gfs_max)
        mp.colorbar()


def draw_map_nogrid_animated(bb, md_range, gfs_dict=None, points=None, marker=None, draw_terrain=False):
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
    mp = Basemap(projection="merc",
                 lat_0=bb[0], lon_0=bb[2],
                 llcrnrlat=bb[0],
                 llcrnrlon=bb[2],
                 urcrnrlat=bb[1],
                 urcrnrlon=bb[3],
                 resolution='i')
    mp.drawcoastlines()
    if draw_terrain:
        mp.etopo()
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
        vals = gfs_dict[(month, day)]
        mp.imshow(vals[plot_bb_0-1:plot_bb_1-1:-1, plot_bb_2:plot_bb_3], vmin=gfs_min, vmax=gfs_max)
        mp.colorbar()

    FuncAnimation(fig, update, frames=np.arange(0, 10), interval=200)