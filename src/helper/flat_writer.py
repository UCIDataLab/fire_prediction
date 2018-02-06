import os
import sys
import datetime as dt
from collections import defaultdict

import numpy as np

import helper.date_util as du
import evaluation.evaluate_model as evm
import helper.weather as weather
import helper.loaders as load

REP_DIR = "/home/cagraff/Documents/dev/fire_prediction/"
SRC_DIR = REP_DIR + 'src/'
DATA_DIR = REP_DIR + 'data/'

def date_to_day_of_year(date):
    return date.year,date.timetuple().tm_yday

ignition_cube_src = os.path.join(DATA_DIR, 'interim/modis/fire_cube/fire_ignition_cube_modis_alaska_2007-2016.pkl')
detection_cube_src = os.path.join(DATA_DIR, 'interim/modis/fire_cube/fire_detection_cube_modis_alaska_2007-2016.pkl')
weather_proc_region_src = os.path.join(DATA_DIR, 'interim/gfs/weather_proc/weather_proc_gfs_4_alaska_2007-2016.pkl')

_, Y_detection_c = evm.setup_ignition_data(ignition_cube_src, detection_cube_src)
Y_detection_c.name = 'num_det'
weather_proc_region = load.load_pickle(weather_proc_region_src)


fill_n_days = 5

t_k = int(sys.argv[2])
print 'T_k=%d' % t_k

def get_date_index(weather_data, target_datetime):
        date_ind = np.searchsorted(weather_data.dates, target_datetime, side='left')

        # Check if left or right element is closer
        if date_ind != 0:
            date_ind_left, date_ind_curr = date_ind-1, date_ind

            dist_left = abs((weather_data.dates[date_ind_left] - target_datetime).total_seconds())
            dist_curr = abs((weather_data.dates[date_ind_curr] - target_datetime).total_seconds())

            if dist_left < dist_curr:
                date_ind = date_ind_left

        return date_ind

def get_weather_variables(vals,weather_data, target_datetime, covariates):                                                             
    # Get date index
    date_ind = get_date_index(weather_data, target_datetime)                                                  

    #vals = []
    for key in covariates:                                                                           
        data = weather_data[key].values                                                                            
        val = data[:, :, date_ind]                                                                 

        if np.any(np.isnan(val)):
            val = fill_missing_value(data, date_ind)                                        

        #vals.append(val)                                                                                           
        vals[key].append(val)

    #return vals                                                                                                    
    
def fill_missing_value(data, date_ind):                                                    
    """
    Try to replace with closest prev day in range [1, fill_n_days].                                                

    If no non-nan value is found, replaces with mean of all values at the given lat/lon.                           
    """ 
    for day_offset in range(1,fill_n_days+1):                                                                 
        new_date_ind = date_ind - day_offset                                                                       

        if new_date_ind < 0:                                                                                       
            break                                                                                                  

        val = data[:, :, new_date_ind]                                                                 

        if not np.any(np.isnan(val)):                                                                                      
            return val

    return np.nanmean(data[:, :, :], axis=2)

vals = defaultdict(list)
for date in Y_detection_c.dates:
    time = 14
    date += du.INC_ONE_DAY * t_k
    # TODO: I think the lon (153) doesn't matter because of the time of day we have  selected 14
    target_datetime = dt.datetime.combine(date, dt.time(time, 0, 0, tzinfo=du.TrulyLocalTzInfo(153, du.round_to_nearest_quarter_hour)))

    get_weather_variables(vals, weather_proc_region, target_datetime, ['temperature','humidity','wind','rain'])

to_flatten = weather.WeatherRegion('flatten')
for k,v in vals.iteritems():
    vals[k] = np.rollaxis(np.array(v), 0, 3)  
    cube = weather.WeatherCube(k, vals[k], None, dates=Y_detection_c.dates)
    to_flatten.add_cube(cube)


# Shift detections by t_k
det_shift = Y_detection_c.values
shape = np.shape(det_shift)[:2]+(t_k,)
det_shift = np.concatenate((det_shift, np.zeros(shape)), axis=2)
det_shift = det_shift[:,:,t_k:]

vals,keys = zip(*[(to_flatten.cubes[k].values,k) for k in ['temperature','humidity','wind','rain']])
vals = (Y_detection_c.values,) + vals + (det_shift,)
keys = ('num_det',) + keys + ('num_det_target',)
to_flatten_arr = np.stack(vals, axis=3)

with open(sys.argv[1], 'wb') as fout:
    header = 'year,day_of_year,row,col,' + ','.join(keys) + '\n'
    fout.write(header)
    for i,d in enumerate(Y_detection_c.dates):
        year,day_of_year = date_to_day_of_year(d)
        for row in range(33):
            for col in range(55):
                line = '%d,%d,%d,%d,%d,%f,%f,%f,%f,%d\n' % ((year,day_of_year, row, col) + tuple(to_flatten_arr[row,col,i,:]))
                fout.write(line)
