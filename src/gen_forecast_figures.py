import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Display simulation
def without_zeros(arr):
    new_arr = [arr[0]]
    for i in range(1, len(arr)):
        ind = arr[i - 1] != 0
        new_arr.append(arr[i][ind])

    return np.array(new_arr)


def display_simulation(pred_dict, observed, with_zeros=True, markers=None):
    x = range(1, len(observed) + 1)

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    ax0 = plt.subplot(gs[0])
    _ = ax0.set_xlabel('Days')

    if with_zeros:
        _ = ax0.set_ylabel('Counts (w/ Zeros)')

        ax0.plot(x, list(map(np.mean, observed)), '--', label='Observed')
        for i, (k, v) in enumerate(pred_dict.items()):
            v, zero = v
            mark = '-'
            if markers is not None:
                mark += markers[i]

            ax0.plot(x, list(map(np.mean, v)), mark, label=k)

    else:
        _ = ax0.set_ylabel('Counts (w/o Zeros)')
        observed_without_zeros = without_zeros(observed)
        ax0.plot(x, list(map(np.mean, observed_without_zeros)), '--', label='Observed')
        for k, v in pred_dict.items():
            ax0.plot(x, list(map(np.mean, without_zeros(v))), label=k)

    _ = ax0.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax1 = plt.subplot(gs[1], sharex=ax0)
    _ = ax1.plot(x, list(map(lambda x: np.sum(x == 0) / len(x), observed)), '--')
    for i, (k, v) in enumerate(pred_dict.items()):
        v, zero = v
        mark = '-'
        if markers is not None:
            mark += markers[i]

        _ = ax1.plot(x, list(map(lambda x: np.mean(x), zero)), mark)

    _ = ax1.set_ylabel('% Zeros')
    _ = ax1.set_xlabel('Days')


def build_forecast_dict(df):
    forecast_dict = {}
    for m in ['Persistence', 'Poisson', 'Hurdle']:
        pred = []
        zero = []
        for i in range(6):
            pred.append(df[m + '_pred_%d' % i].values)
            zero.append(df[m + '_zeros_%d' % i].values)

        forecast_dict[m] = (pred, zero)

    observed = []
    for i in range(6):
        observed.append(df['num_det_%d' % i].values)

    return forecast_dict, observed


in_file = sys.argv[1]
df = pd.read_csv(in_file)

forecast_dict, observed = build_forecast_dict(df)

display_simulation(forecast_dict, observed, with_zeros=True, markers=['*', '^', 's'])
plt.show()
