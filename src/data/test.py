import sys
import numpy as np

import pygrib
import grib
from helper.geometry import get_default_bounding_box

def foo1(file_name):
    grib = pygrib.open(file_name)
    bb = (55, 71, -165, -138)

    print 'bb', bb

    lats, lons = grib.select(name='Temperature', typeOfLevel='surface')[0].latlons()
    print lats
    print lons
    gfs_bb_0 = np.where(lats[:,0] <= bb[1])[0][0]
    gfs_bb_1 = np.where(lats[:,0] >= bb[0])[0][-1]+1
    gfs_bb_2 = np.where(lons[0,:] >= (bb[2] % 360))[0][0]
    gfs_bb_3 = np.where(lons[0,:] <= (bb[3] % 360))[0][-1]+1
    layer = grib.select(name='Temperature', typeOfLevel='surface')[0].values
    layer = layer[gfs_bb_0:gfs_bb_1, gfs_bb_2:gfs_bb_3]

    dlats = lats[gfs_bb_0:gfs_bb_1, gfs_bb_2:gfs_bb_3]
    dlons = lons[gfs_bb_0:gfs_bb_1, gfs_bb_2:gfs_bb_3]

    return layer, dlats, dlons 

def foo2(file_name):
    bb = get_default_bounding_box()
    selections = grib.get_default_selections()
    with open(file_name) as fin:
        f = grib.GribFile(fin)
        sel = grib.GribSelector(selections, bb)
        extracted = sel.select(f)

    return extracted



l1, dlats, dlons = foo1(sys.argv[1])
print np.shape(l1), l1
print dlats
print dlons-360

l2 = foo2(sys.argv[1])
l2 = l2['Temperature']['values']
print np.shape(l2), l2

print l1==l2


