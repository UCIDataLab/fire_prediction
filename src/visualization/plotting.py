"""
Generating plots.
"""

import matplotlib.pyplot as plt
from evaluation import metrics

from collections import defaultdict

def flat(x):
    return map(lambda x: x.flatten(), x)

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

def plot_results(results, t_k_arr):
    for metric in [metrics.mean_absolute_error, metrics.root_mean_squared_error, metrics.neg_log_likelihood_poisson]:
        for k,v in results.iteritems():
            plt.plot(t_k_arr, map(lambda x: metric(*flat(x)), results[k]), "s--", label=k, linewidth=2)
        lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel("Day of forecast (k)")
        plt.xticks(t_k_arr)
        plt.ylabel(metric.__name__)
        plt.show()

def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)

def plot_results_grid(results_list, t_k_arr, metrics_):
    #fig = plt.figure()
    f, axs = plt.subplots(len(metrics_), len(results_list), sharey='row')
    out = recursive_defaultdict()
    for j,(results,t) in enumerate(results_list):
        for i, metric in enumerate(metrics_):    
            ax = axs[i,j] if len(results_list) > 1 else axs[i] if len(metrics_) > 1 else axs
            #ax = plt.subplot(len(metrics_),len(results_list),(i*len(results_list))+j+1)
            ax.set_title(t)
            for k,v in sorted(results.items()):     
                x = range(1,len(results[k])+1)
                y = map(lambda x: metric(*flat(x)), results[k])
                ax.plot(x, y, "s--", label=k, linewidth=2)

                out[t][k][metric.__name__] = (x,y)
            ax.set_xlabel("Day of forecast (k)")
            ax.set_xticks(t_k_arr)
            if j == 0:
                ax.set_ylabel(metric.__name__)
            if i == 0:
                ax.legend(loc=0)
    return out
            
    #lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

def plot_results_grid_errorbars(results_list, t_k_arr, metrics_):
    #fig = plt.figure()
    f, axs = plt.subplots(len(metrics_), len(results_list), sharey='row')
    out = recursive_defaultdict()
    for j,(results,t) in enumerate(results_list):
        for i, metric in enumerate(metrics_):    
            ax = axs[i,j] if len(results_list) > 1 else axs[i] if len(metrics_) > 1 else axs
            #ax = plt.subplot(len(metrics_),len(results_list),(i*len(results_list))+j+1)
            ax.set_title(t)
            for k,v in sorted(results.items()):     
                x = range(1,len(results[k])+1)
                y_final = []
                for t_k in range(0,5):
                    y = map(lambda x: metric(*flat(x)), results[k][t_k])
                    y_final.append(y)
                error_bars = map(lambda x: np.std(x), y_final)
                print error_bars
                print y_final
                y = map(lambda x: np.mean(np.power(x,2)), y_final)
                ax.errorbar(x, y, yerr=None, fmt="s--", label=k, linewidth=2)

                out[t][k][metric.__name__] = (x,y)
            ax.set_xlabel("Day of forecast (k)")
            ax.set_xticks(t_k_arr)
            if j == 0:
                ax.set_ylabel(metric.__name__)
            if i == 0:
                ax.legend(loc=0)
    return out
 
