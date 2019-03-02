from numpy import pi, cos

t = 1111950.  # size of grid
rho = 6371007.181  # radius of the earth


def tile2latlon(h, v, i, j, n_pix=2400):
    """ Convert burnt data tile coordinates into latitude and longitude
    :param h: horizontal position of tile
    :param v: vertical position of tile
    :param i: column within tile
    :param j: row within tile
    :param n_pix: size of a tile in pixels
    :return: (lat,lon) of the point defined by h, v, i, and j
    """
    x = ((i + .5) * (t / n_pix)) + h * t
    y = ((9 - v) * t) - ((j + .5) * (t / n_pix))
    lat = (y * 180) / (rho * pi)
    lon = (x * 180) / (rho * pi * cos(lat * pi / 180))
    return lat, lon


def latlon2tile(lat, lon, n_pix=2400):
    x = rho * cos(lat * (pi / 180)) * lon * (pi / 180)
    y = rho * lat * (pi / 180)
    h = int(x / t) + 18
    v = 8 - int(y / t)
    return h, v
