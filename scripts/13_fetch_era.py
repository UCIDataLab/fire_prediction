"""
@Shane Coffield
from ERA website
scoffiel@uci.edu
Purpose: batch download ERA-interim weather data
time to run ~ 15 minutes
'''

'''prerequisites
conda install -c conda-forge ecmwf-api-client
retrieve your key at https://api.ecmwf.int/v1/key/
create dot file under /Users/scoffiel/.ecmwfapirc with the following contents
{
    "url"   : "https://api.ecmwf.int/v1",
    "key"   : "191269340f4ad15229c2cb135baf5802",
    "email" : "scoffiel@uci.edu"
}

"""

# noinspection PyUnresolvedReferences
from ecmwfapi import ECMWFDataServer
import xarray as xr
import numpy as np

root = '/Users/scoffiel/fire_prediction/era_interim/'

years = range(2007, 2017)

for yr in years:
    # precip
    server = ECMWFDataServer()

    server.retrieve({
        "class": "ei",
        "dataset": "interim",
        "date": "{}-01-01/to/{}-12-31".format(yr, yr),
        "expver": "1",
        "grid": "0.75/0.75",
        "levtype": "sfc",
        "param": "228.128",  # total precip for the future 12 hours
        "step": "12",  # 3/6/9/
        "stream": "oper",
        "time": "00:00:00/12:00:00",
        "type": "fc",
        "area": "71/-165/55/-138",  # N/W/S/E
        "format": "netcdf",
        "target": "{}{}_precip.nc".format(root, yr)
    })

    del server

    # other variables
    server = ECMWFDataServer()

    server.retrieve({
        "class": "ei",
        "dataset": "interim",
        "date": "{}-01-01/to/{}-12-31".format(yr, yr),
        "grid": "0.75/0.75",
        "levtype": "sfc",
        "param": "165.128/166.128/167.128/168.128",
        "step": "0",
        "stream": "oper",
        "time": "0/6/12/18",
        "type": "an",
        "area": "71/-165/55/-138",  # N/W/S/E
        "format": "netcdf",
        "target": "{}{}_others.nc".format(root, yr)
    })

    del server

    ds = xr.open_dataset(root + '{}_others.nc'.format(yr))
    ws = xr.DataArray(np.sqrt(ds.u10 ** 2 + ds.v10 ** 2))
    ws.name = 'ws'
    ds2 = ws.to_dataset()
    ds3 = xr.merge([ds, ds2])
    ds3.to_netcdf(root + '{}.nc'.format(yr))
