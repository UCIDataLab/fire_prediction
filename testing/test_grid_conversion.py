from geometry.grid_conversion import get_latlon_xy_fxns, ak_bb, get_gfs_val
import numpy as np
import cPickle

TOL = 1E-3


def test_get_latlon_xy_fxns():
    x = 388.876
    y = 555
    lat = 60
    lon = -160

    latlon2xy, xy2latlon, bb_shape = get_latlon_xy_fxns(ak_bb)
    assert(np.abs(bb_shape[1] - 1776) <= TOL)
    assert(np.abs(bb_shape[0] - 1723.980) <= TOL)

    x_hat,y_hat = latlon2xy(lat,lon)
    print (x_hat, y_hat)
    assert(np.abs(x - x_hat) <= TOL)
    assert(np.abs(y - y_hat) <= TOL)

    lat_hat,lon_hat = xy2latlon(x,y)
    assert(np.abs(lat - lat_hat) <= TOL)
    assert(np.abs(lon - lon_hat) <= TOL)


#def test_get_latlon_grid_fxns():


def test_get_gfs_val(test_dict_loc="data/temp_dict.pkl"):
    with open(test_dict_loc) as fpkl:
        gfs_dict = cPickle.load(fpkl)
    lat = 89.5
    lon = 50.0
    val = get_gfs_val(lat, lon, 0, gfs_dict)
    assert(np.abs(gfs_dict[(1,1)][1,100] - val) <= TOL)
    val2 = get_gfs_val(lat-.1, lon-.1, 0, gfs_dict)
    assert(np.abs(gfs_dict[(1,1)][1,100] - val2) <= TOL)


if __name__ == "__main__":
    test_get_latlon_xy_fxns()
    test_get_gfs_val()
    print "All tests passed!"
