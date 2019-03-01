""" Store pipeline parameters used in multiple stages of the pipeline. """

"""
Define the different options for measurement selection.
"""
import numpy as np

from data import grib
from helper.geometry import LatLonBoundingBox


# === General ===
# Region bounding boxes
bounding_boxes = {}
bounding_boxes['alaska'] = LatLonBoundingBox(55, 71, -165, -138)
bounding_boxes['global'] = LatLonBoundingBox(-90, 90, -180, 180)

REGION_BOUNDING_BOXES = bounding_boxes

# === Directories ===
GFS_RAW_DATA_DIR = 'raw/gfs'
GFS_FILTERED_DATA_DIR = 'interim/gfs/filtered'
GFS_AGGREGATED_DATA_DIR = 'interim/gfs/aggregated'
GFS_REGION_DATA_DIR = 'interim/gfs/region'

MODIS_RAW_DATA_DIR = 'raw/modis/'
MODIS_AGGREGATED_DATA_DIR = 'interim/modis/aggregated'
MODIS_REGION_DATA_DIR = 'interim/modis/region'

# === Server Info ===
GFS_SERVER_NAME = 'nomads.ncdc.noaa.gov'  # Server from which to pull the GFS data
GFS_SERVER_USERNAME = 'anonymous'
GFS_SERVER_PASSWORD = 'graffc@uci.edu'
GFS_SERVER_DATA_DIR = "GFS/analysis_only/"  # location on server of GFS data

MODIS_SERVER_NAME = 'fuoco.geog.umd.edu'
MODIS_SERVER_USERNAME = 'fire'
MODIS_SERVER_PASSWORD = 'burnt'
MODIS_SERVER_DATA_DIR = 'modis/C6/mcd14ml'

# === GFS ===
GFS_RESOLUTIONS = ['3', '4']
GFS_TIMES = [0, 6, 12, 18]
GFS_OFFSETS = [0, 3, 6]

# === ERA ===
ERA_RESOLUTIONS = ['4']
ERA_TIMES = [0,6,12,18]
ERA_OFFSETS = [0]

# GFS Measurement Selections
temperature = grib.GribSelection('temperature', np.float32).add_selection(name='Temperature', typeOfLevel='surface')
humidity = grib.GribSelection('humidity', np.float32).add_selection(name='Surface air relative humidity').add_selection(name='2 metre relative humidity').add_selection(name='Relative humidity', level=2)
wind_u = grib.GribSelection('u_wind_component', np.float32).add_selection(name='10 metre U wind component')
wind_v = grib.GribSelection('v_wind_component', np.float32).add_selection(name='10 metre V wind component')
rain = grib.GribSelection('precipitation', np.float32).add_selection(name='Total Precipitation')

cape0 = grib.GribSelection('cape_0', np.int16).add_selection(name='Convective available potential energy', level=0)
cape18000 = grib.GribSelection('cape_18000', np.int16).add_selection(name='Convective available potential energy', level=18000)

pblh = grib.GribSelection('pbl_height', np.float32).add_selection(name='Planetary boundary layer height')
cloud = grib.GribSelection('total_cloud_cover', np.float32).add_selection(name='Total Cloud Cover')
soilm = grib.GribSelection('soil_moisture_content', np.float32).add_selection(name='Volumetric soil moisture content', level=0)
mask = grib.GribSelection('land_sea_mask', np.bool_).add_selection(name='Land-sea mask')
orog = grib.GribSelection('orography', np.float32).add_selection(name='Orography')
shortwrad = grib.GribSelection('short_wave_rad_flux', np.float32).add_selection(name='Downward short-wave radiation flux')

selections = {}
selections['default_v1'] = [temperature, humidity, wind_u, wind_v, rain, cape0, cape18000, pblh, cloud, soilm, mask, orog, shortwrad]

GFS_MEASUREMENT_SEL = selections

selections_era = {}
selections_era['default_v1'] = [temperature, humidity, wind_u, wind_v, rain]
ERA_MEASUREMENT_SEL = selections_era

# === Weather ===
WEATHER_FILL_METH = ['integrate_mean', 'integrate_interp', 'mean', 'interpolate', 'drop']
