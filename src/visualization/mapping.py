"""
Generating and modifying spatial maps.
"""

from mpl_toolkits.basemap import Basemap

def make_map(bb):
    lat_min, lat_max, lon_min, lon_max = bb.get()

    mp = Basemap(projection="merc",
                  llcrnrlat=lat_min,
                  llcrnrlon=lon_min,
                  urcrnrlat=lat_max,
                  urcrnrlon=lon_max,
                  resolution='i')

    mp.drawcoastlines()
    #mp.drawlsmask()

    parallels = np.arange(lat_min,lat_max,2)
    _ = mp.drawparallels(parallels,labels=[False,True,False,False])
    
    parallels = np.arange(lat_min,lat_max,.5)
    _ = mp.drawparallels(parallels,labels=[False,False,False,False])

    meridians = np.arange(lon_min,lon_max,2)
    _ = mp.drawmeridians(meridians, labels=[False,False,False,True])
    
    meridians = np.arange(lon_min,lon_max,.5)
    _ = mp.drawmeridians(meridians, labels=[False,False,False,False])
    
    return mp
