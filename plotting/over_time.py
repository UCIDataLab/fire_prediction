import matplotlib.pyplot as plt
import numpy as np
import cPickle
from geometry.grid_conversion import small_fire_bb
from util.daymonth import monthday2day, increment_day
from mpl_toolkits.basemap import Basemap


def fire_hist(modis, outfi='pics/fire_hist.png', title="Total fire detections per month"):
    years = (np.min(modis.year), np.max(modis.year))
    total_fires_arr = []
    month_float_arr = []
    year = years[0]
    month = np.min(modis[modis.year==years[0]].month)
    while year < years[1] or month < np.max(modis[modis.year==years[1]].month):
        month_float = year + (float(month-1) / 12.)
        total_fires = len(modis[(modis.year==year) & (modis.month==month)])
        month_float_arr.append(month_float)
        total_fires_arr.append(total_fires)
        month += 1
        if month == 13:
            month = 1
            year += 1
    plt.plot(month_float_arr, total_fires_arr, 'r')
    plt.title(title + " (%d-%d)" % years)
    plt.savefig(outfi)
    plt.close()


def color_fire_by_time(modis, fire_bb, year, outfi='pics/small_fire_over_time'):
    my_fires = modis[(modis.lat < fire_bb[1]) & (modis.lat > fire_bb[0]) &
                     (modis.lon < fire_bb[3]) & (modis.lon > fire_bb[2]) & (modis.year == year)]
    min_month = np.min(my_fires.month)
    min_dayy = np.min(my_fires[my_fires.month == min_month].day)
    min_day = monthday2day(min_month, min_dayy, leapyear=(year%4)==0)
    max_month = np.max(my_fires.month)
    max_dayy = np.max(my_fires[my_fires.month == min_month].day)
    max_day = monthday2day(max_month, max_dayy, leapyear=(year%4)==0)

    lats = []
    longs = []
    colors = []
    month = min_month
    day = min_dayy
    while month < max_month or day < max_dayy:
        dayofyear = monthday2day(month, day, leapyear=(year%4)==0)
        todays_fires = my_fires[(my_fires.month == month) & (my_fires.day == day)]
        today_longs, today_lats = np.array(todays_fires.lon), np.array(todays_fires.lat)
        lats += list(today_lats)
        longs += list(today_longs)
        colors += [float(dayofyear)]*len(todays_fires) #/ (max_day - min_day)]*len(todays_fires)
        year, month, day = increment_day(year, month, day)
    plt.scatter(longs, lats, c=colors)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.colorbar()
    plt.title("%d Alaska fire over time (color=day of year)" %year)
    plt.savefig(outfi)


if __name__ == "__main__":
    with open('data/ak_fires.pkl') as fmod:
        modis = cPickle.load(fmod)
    #fire_hist(modis, 'pics/fire_hist_ak.png', "Fire detections per month (Alaska)")
    #with open('data/full_modis.pkl') as fmod:
    #    modis = cPickle.load(fmod)
    #fire_hist(modis, 'pics/fire_hist_full.png', "Fire detections per month (World)")
    color_fire_by_time(modis, small_fire_bb, 2013)
