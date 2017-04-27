from grid_conversion import get_latlon_xy_fxns
from grid_conversion import ak_bb
import pandas as pd


def append_xy(df, bb):
    latlon2xy, _, shape = get_latlon_xy_fxns(bb)
    valid_fires = df[(df.lat < bb[1]) & (df.lat > bb[0]) &
                     (df.long < bb[3]) & (df.long > bb[2])]
    xys = map(latlon2xy, valid_fires.lat, valid_fires.long)
    xs = map(lambda x: x[0], xys)
    ys = map(lambda x: x[1], xys)
    valid_fires.loc[:,'x'] = xs
    valid_fires.loc[:,'y'] = ys
    return valid_fires
