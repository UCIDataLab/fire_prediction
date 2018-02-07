"""
Generating plots.
"""

import matplotlib.pyplot as plt
from evaluation import metrics

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
    for metric in [metrics.mean_absolute_error, metrics.root_mean_squared_error]:
        for k,v in results.iteritems():
            plt.plot(t_k_arr, map(lambda x: metric(*flat(x)), results[k]), "s--", label=k, linewidth=2)
        lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel("Day of forecast (k)")
        plt.xticks(t_k_arr)
        plt.ylabel(metric.__name__)
        plt.show()

def plot_results_grid(results_list, t_k_arr):
    #fig = plt.figure()
    metrics_ = [metrics.mean_absolute_error, metrics.root_mean_squared_error]
    f, axs = plt.subplots(len(metrics_), len(results_list), sharey='row')
    for j,(results,t) in enumerate(results_list):
        for i, metric in enumerate(metrics_):    
            ax = axs[i,j]
            #ax = plt.subplot(len(metrics_),len(results_list),(i*len(results_list))+j+1)
            ax.set_title(t)
            for k,v in results.iteritems():     
                ax.plot(range(1,len(results[k])+1), map(lambda x: metric(*flat(x)), results[k]), "s--", label=k, linewidth=2)
            ax.set_xlabel("Day of forecast (k)")
            ax.set_xticks(t_k_arr)
            axs[i,0].set_ylabel(metric.__name__)
            
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
