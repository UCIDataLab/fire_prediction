"""
Generating plots.
"""

import matplotlib.pyplot as plt
from models import metrics

def plot_training(results, t_k_arr, metric=metrics.mean_absolute_error):
    plt.plot(t_k_arr+1, map(lambda x: metric(*x), results['baseline']), "kv--", label="Baseline", linewidth=2)
    plt.plot(t_k_arr+1, map(lambda x: metric(*x), results['auto']), "gs--", label="Autoregression", linewidth=2)
    plt.plot(t_k_arr+1, map(lambda x: metric(*x), results['temp_humid']), "r^--", label="Temp/hum", linewidth=2)
    plt.plot(t_k_arr+1, map(lambda x: metric(*x), results['all']), "bo--", label="All weather", linewidth=2)
    #plt.plot(t_k_arr+1, map(lambda x: metric(*x), results['auto_linear']), "cs--", label='Linear Autoregression', linewidth=2)
    #plt.plot(t_k_arr+1, map(lambda x: metric(*x), results['linear_all']), "yo--", label='Linear All', linewidth=2)

    plt.rcParams.update({'font.size': 14})
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel("Day of forecast (k)")
    plt.xticks(t_k_arr+1)
    plt.ylabel(metric.__name__)
