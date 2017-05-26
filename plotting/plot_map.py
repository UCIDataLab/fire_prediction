import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.basemap import Basemap
import os
from util.daymonth import day2monthday, monthday2day
import numpy as np


def draw_map_nogrid_static(bb, month=1, day=1, year=None, gfs_dict=None, latlon_df=None, marker='ro', draw_terrain=False,
                           title=None):
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
    if latlon_df is not None:
        mp_lons, mp_lats = mp(np.array(latlon_df.lon), np.array(latlon_df.lat))
        mp.plot(mp_lons, mp_lats, marker)
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
        if year is not None:
            vals = gfs_dict[(year, month, day)]
        else:
            vals = gfs_dict[(month, day)]
        mp.imshow(vals[plot_bb_0-1:plot_bb_1-1:-1, plot_bb_2:plot_bb_3], vmin=gfs_min, vmax=gfs_max)
        mp.colorbar()
    if title:
        plt.title(title)


def make_gif(df, gfs_dict, name="Temp"):
    ak_bb = [55,71,-165,-138]
    lats = gfs_dict['lats']
    lons = gfs_dict['lons']
    plot_bb_0 = np.where(lats[:,0] <= ak_bb[0])[0][0]
    plot_bb_1 = np.where(lats[:,0] <= ak_bb[1])[0][0]
    plot_bb_2 = np.where(lons[0,:] >= (ak_bb[2] % 360))[0][0]
    plot_bb_3 = np.where(lons[0,:] >= (ak_bb[3] % 360))[0][0]

    mp = Basemap(projection="merc",
                  lat_0=55, lon_0=-165,
                  llcrnrlat=55,
                  llcrnrlon=-165,
                  urcrnrlat=71,
                  urcrnrlon=-138,
                  resolution='i')
    start_day = monthday2day(6,1)
    end_day = monthday2day(9,1)
    min_temp = gfs_dict['min']
    max_temp = gfs_dict['max']
    prev_lats = []
    prev_lons = []
    for dayy in xrange(start_day, end_day):
        if len(prev_lats):
            mp.plot(np.array(prev_lons), np.array(prev_lats), 'ko')
        monthday = day2monthday(dayy)
        today_fires = df[(df.year == 2013) & (df.month == monthday[0]) & (df.day == monthday[1])]
        if len(today_fires):
            mp_lons, mp_lats = mp(np.array(today_fires.lon), np.array(today_fires.lat))
            mp.plot(mp_lons, mp_lats, 'ro')
            prev_lats += list(mp_lats)
            prev_lons += list(mp_lons)
        temp_vals = gfs_dict[monthday]
        mp.imshow(temp_vals[plot_bb_0-1:plot_bb_1-1:-1, plot_bb_2:plot_bb_3],
                  vmin=min_temp, vmax=max_temp)
        plt.title("%s for %d/%d" % (name, monthday[0], monthday[1]))
        mp.drawcoastlines()
        mp.colorbar()
        plt.savefig('gifmaking/day%d.png' % dayy)
        plt.close()

    os.system('convert -delay 100 -loop 0 gifmaking/day*.png gifmaking/%s_loop_2013.gif' % name)


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