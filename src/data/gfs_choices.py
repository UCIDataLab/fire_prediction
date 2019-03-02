"""
Define the different options for measurement selection.
"""
import numpy as np
from helper.geometry import LatLonBoundingBox

from .grib import GribSelection


def build_bounding_boxes():
    bounding_boxes = {'alaska': LatLonBoundingBox(55, 71, -165, -138), 'global': LatLonBoundingBox(-90, 90, -180, 180)}

    return bounding_boxes


GFS_BOUNDING_BOXES = build_bounding_boxes()


def build_measurement_sel():
    """
    Build a list of GribSelections for the default GFS measurements used.
    """
    selections = {}

    temperature = GribSelection('temperature', np.float32).add_selection(name='Temperature', typeOfLevel='surface')
    humidity = GribSelection('humidity', np.uint8).add_selection(name='Surface air relative humidity').add_selection(
        name='2 metre relative humidity').add_selection(name='Relative humidity', level=2)
    wind_u = GribSelection('u_wind_component', np.float32).add_selection(name='10 metre U wind component')
    wind_v = GribSelection('v_wind_component', np.float32).add_selection(name='10 metre V wind component')
    rain = GribSelection('precipitation', np.float32).add_selection(name='Total Precipitation')

    cape0 = GribSelection('cape_0', np.int16).add_selection(name='Convective available potential energy', level=0)
    cape18000 = GribSelection('cape_18000', np.int16).add_selection(name='Convective available potential energy',
                                                                    level=18000)
    # cape25500 = GribSelection('cape_25500').add_selection(name='Convective available potential energy', level=25500)

    pblh = GribSelection('pbl_height', np.float32).add_selection(name='Planetary boundary layer height')
    cloud = GribSelection('total_cloud_cover', np.uint8).add_selection(name='Total Cloud Cover')
    soilm = GribSelection('soil_moisture_content', np.float32).add_selection(name='Volumetric soil moisture content',
                                                                             level=0)
    mask = GribSelection('land_sea_mask', np.bool_).add_selection(name='Land-sea mask')
    orog = GribSelection('orography', np.float32).add_selection(name='Orography')
    shortwrad = GribSelection('short_wave_rad_flux', np.float32).add_selection(
        name='Downward short-wave radiation flux')

    selections['default_v1'] = [temperature, humidity, wind_u, wind_v, rain, cape0, cape18000, pblh, cloud, soilm, mask,
                                orog, shortwrad]

    return selections


GFS_MEASUREMENT_SEL = build_measurement_sel()
