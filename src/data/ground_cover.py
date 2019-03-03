import pickle

import numpy as np

with open('data/raw/land_mcd12c1/land_cover.pkl', 'rb') as fin:
    data = pickle.load(fin)

num_land_types = len(set(data.flatten()))
land_cover_types = np.zeros((data.shape[0] / 10, data.shape[1] / 10, num_land_types))

for i in range(0, data.shape[0]):
    for j in range(0, data.shape[1]):
        land_cover_types[i / 10, j / 10, data[i, j]] += 1

lat_max_ind = (90 - 71) * 2
lat_min_ind = (90 - 55) * 2

lon_max_ind = (180 - 165) * 2
lon_min_ind = (180 - 138) * 2

sel_land_cover_types = land_cover_types[lat_max_ind:lat_min_ind + 1, lon_max_ind:lon_min_ind + 1]

with open('land_cover_alaska.pkl', 'wb') as f_out:
    pickle.dump(sel_land_cover_types, f_out, protocol=2)
