"""
Generating and modifying spatial maps.
"""

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.basemap import Basemap

from src.helper import date_util as du


def make_map(bb, grid_spacing=(2, .5, 2, .5)):
    lat_min, lat_max, lon_min, lon_max = bb.get()

    mp = Basemap(projection="merc",
                 llcrnrlat=lat_min,
                 llcrnrlon=lon_min,
                 urcrnrlat=lat_max,
                 urcrnrlon=lon_max,
                 resolution='i')

    mp.drawcoastlines()
    # mp.drawlsmask()

    if grid_spacing[0] != 0:
        parallels = np.arange(lat_min, lat_max, grid_spacing[0])
        _ = mp.drawparallels(parallels, labels=[False, True, False, False])

    if grid_spacing[1] != 0:
        parallels = np.arange(lat_min, lat_max, grid_spacing[1])
        _ = mp.drawparallels(parallels, labels=[False, False, False, False])

    if grid_spacing[2] != 0:
        meridians = np.arange(lon_min, lon_max, grid_spacing[2])
        _ = mp.drawmeridians(meridians, labels=[False, False, False, True])

    if grid_spacing[3] != 0:
        meridians = np.arange(lon_min, lon_max, grid_spacing[3])
        _ = mp.drawmeridians(meridians, labels=[False, False, False, False])

    return mp


def make_contourf(mp, values, bb, alpha=.6, vmin=None, vmax=None):
    mp.shadedrelief()

    lats, lons = bb.make_grid()

    cs = mp.contourf(lons, lats, values, latlon=True, alpha=alpha, vmin=vmin, vmax=vmax)
    _ = mp.colorbar(cs, location='bottom', pad="5%")


def animate_map_latlon(df, bb, dates, figsize=(10, 12)):
    fig = plt.figure(figsize=figsize)

    mp = make_map(bb)
    mp.shadedrelief()

    s2 = mp.scatter([], [], 30, latlon=True, marker='o', color='b', alpha=.7)
    s = mp.scatter([], [], 30, latlon=True, marker='o', color='r', alpha=.7)

    def init():
        s.set_offsets([])
        s2.set_offsets([])
        return s, s2

    def animate(i):
        date = dates[i]

        _ = plt.title('Date %s (day %d)' % (str(date), du.day_of_year_from_datetime(date)))

        sel_df = df[df.date_local == date]

        lons, lats = mp(list(sel_df.lon), list(sel_df.lat))
        s.set_offsets(zip(lons, lats))

        sel_df = df[df.date_local < date]
        lons, lats = mp(list(sel_df.lon), list(sel_df.lat))
        s2.set_offsets(zip(lons, lats))

        return s, s2

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(dates), interval=1000, blit=True)

    return anim


def animate_map_latlon_all(df, cluster_id, bb, dates, show_prev=True, figsize=(10, 12)):
    fig = plt.figure(figsize=figsize)

    cluster_df = df[df.cluster_id == cluster_id]
    non_df = df[(df.cluster_id != cluster_id) & (df.date_local >= np.min(cluster_df.date_local))]

    if show_prev:
        year = np.min(cluster_df.date_local.year)
        prev_df = df[(df.cluster_id != cluster_id) & (df.date_local < np.min(cluster_df.date_local)) & (
                df.date_local >= dt.date(year, 1, 1))]

    mp = make_map(bb)
    mp.shadedrelief()

    if show_prev:
        s5 = mp.scatter([], [], 10, latlon=True, marker='o', color='m', alpha=.7)
    s4 = mp.scatter([], [], 10, latlon=True, marker='o', color='g', alpha=.7)
    s3 = mp.scatter([], [], 10, latlon=True, marker='o', color='y', alpha=.7)
    s2 = mp.scatter([], [], 10, latlon=True, marker='o', color='b', alpha=.7)
    s = mp.scatter([], [], 10, latlon=True, marker='o', color='r', alpha=.7)

    def init():
        s.set_offsets([])
        s2.set_offsets([])
        return s, s2

    def animate(i):
        date = dates[i]

        _ = plt.title('Date %s (day %d)' % (str(date), du.day_of_year_from_datetime(date)))

        sel_df = cluster_df[cluster_df.date_local == date]

        lons, lats = mp(list(sel_df.lon), list(sel_df.lat))
        s.set_offsets(zip(lons, lats))

        sel_df = cluster_df[cluster_df.date_local < date]
        lons, lats = mp(list(sel_df.lon), list(sel_df.lat))
        s2.set_offsets(zip(lons, lats))

        sel_df = non_df[non_df.date_local == date]
        lons, lats = mp(list(sel_df.lon), list(sel_df.lat))
        s3.set_offsets(zip(lons, lats))

        sel_df = non_df[non_df.date_local < date]
        lons, lats = mp(list(sel_df.lon), list(sel_df.lat))
        s4.set_offsets(zip(lons, lats))

        if show_prev:
            sel_df = prev_df
            lons, lats = mp(list(sel_df.lon), list(sel_df.lat))
            s5.set_offsets(zip(lons, lats))

        return s, s2

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(dates), interval=1000, blit=True)

    return anim
