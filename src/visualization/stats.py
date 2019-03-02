"""
Generating statistics for spatial/temporal data.
"""

import numpy as np
from scipy.stats.stats import pearsonr


def calc_mean(values, shape):
    table = []
    mean = np.zeros(shape[:2])
    for lat in range(0, shape[0]):
        for lon in range(0, shape[1]):
            v = values[lat, lon]

            # Remove nans
            v = v[np.logical_not(np.isnan(v))]

            mean[lat, lon] = np.mean(v)
    return mean


def calc_cor(values, shape, lat_off_tup, lon_off_tup):
    min_lat_off, max_lat_off, lat_off = lat_off_tup
    min_lon_off, max_lon_off, lon_off = lon_off_tup
    table = []
    cor = np.zeros(shape[:2])
    for lat in range(min_lat_off, shape[0] + max_lat_off):
        for lon in range(min_lon_off, shape[1] + max_lon_off):
            v = values[lat, lon]
            v_off = values[lat + lat_off, lon + lon_off]

            # Remove nans
            v = v[np.logical_not(np.isnan(v))]
            v_off = v_off[np.logical_not(np.isnan(v_off))]

            cor[lat, lon] = pearsonr(v, v_off)[0]
    return cor
