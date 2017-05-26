import matplotlib as plt
import numpy as np


total_fires_arr = np.zeros(n_months)
month_float_arr = np.zeros(n_months)
for i,dl in enumerate(data_list_list):
    year = dl[0][0]
    month = dl[0][1]
    month_float = year + (float(month-1) / 12.)
    total_fires = len(dl)
    month_float_arr[i] = month_float
    total_fires_arr[i] = total_fires
plt.plot(month_float_arr, total_fires_arr, 'r')
plt.title("Total fires per month (%d-%d)" %(int(math.floor(np.min(month_float_arr))), int(math.floor(np.max(month_float_arr)))))
plt.show()