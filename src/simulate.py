import pickle
from pipeline.train_pipeline import setup_data
import datetime as dt

fn = '/lv_scratch/scratch/graffc0/fire_prediction/experiments/280924225779574237060668310148955903472.pkl'
file_path_fmt = '/lv_scratch/scratch/graffc0/fire_prediction/data/processed/grid/grid_ds_gfs_4_modis_alaska_2007-01-01_2016-12-31_integrate_interp_%dk.nc'

# Load model/params
with open(fn, 'rb') as fin:
    save = pickle.load(fin)

model = save['models']
params = save['params']

# Setup data
in_files = {k: file_path_fmt % k for k in range(1,5+1)}
start_date = dt.date(2007, 1, 1)
end_date = dt.date(2016, 12, 31)
forecast_horizon = params['forecast_horizon']

X_grid_dict_nw, y_grid_dict, years_train = setup_data(in_files, start_date, end_date, forecast_horizon, params)

def simulate_day(params, model):
    pass
