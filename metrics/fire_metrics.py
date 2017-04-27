import numpy as np
from scipy.spatial import ConvexHull
from util.daymonth import monthday2day


def get_summary_stats(df, clusts, n_CCs):
    # Summary stats we want to collect
    len_arr = np.zeros(n_CCs)   # Total detections in fire
    hull_area_arr = np.zeros(n_CCs)   # Area of the convex hull of detections
    mean_dist_from_center_arr = np.zeros(n_CCs)   # Mean distance from centroid of detections
    fire_duration_arr = np.zeros(n_CCs)   # How long each fire lasts

    for clust in xrange(n_CCs):
        clust_fires = df.iloc[np.where(clusts == clust)]
        len_arr[clust] = len(clust_fires)
        mean_x = np.mean(clust_fires.x)
        mean_y = np.mean(clust_fires.y)
        dist_from_center_arr = []
        for y,x in zip(clust_fires.y, clust_fires.x):
            dist_from_center_arr.append(np.sqrt((x - mean_x)**2 + (y - mean_y)**2))
        xy_mat = np.column_stack((clust_fires.x, clust_fires.y))
        if len(clust_fires) >= 3:
            hull_area_arr[clust] = ConvexHull(xy_mat).volume
        else:
            hull_area_arr[clust] = 0
        mean_dist_from_center_arr[clust] = np.mean(dist_from_center_arr)

        min_day = np.inf
        max_day = -np.inf
        for i,(month,day) in enumerate(zip(clust_fires.month, clust_fires.day)):
            my_day = monthday2day(month, day, leapyear=False)
            if my_day < min_day:
                min_day = my_day
            if my_day > max_day:
                max_day = my_day
        fire_duration_arr[clust] = max_day - min_day

    ret_dict = dict()
    ret_dict['len'] = len_arr
    ret_dict['area'] = hull_area_arr
    ret_dict['dist_from_center'] = mean_dist_from_center_arr
    ret_dict['duration'] = fire_duration_arr
    return ret_dict


def get_time_series(df, clusts, n_CCs, zero_centered=False):
    time_series = []
    for clust in xrange(n_CCs):
        clust_fires = df.iloc[np.where(clusts == clust)]
        time_arr = np.zeros(len(clust_fires))
        for i,(month,day) in enumerate(zip(clust_fires.month, clust_fires.day)):
            my_day = monthday2day(month, day, leapyear=False)
            time_arr[i] = my_day
        sorted_times = np.sort(time_arr)
        if zero_centered:
            min_day = sorted_times[0]
            time_series.append(sorted_times - min_day)
        else:
            time_series.append(sorted_times)
    return time_series

