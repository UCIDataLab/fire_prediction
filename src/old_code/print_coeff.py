"""
Print coefficients from saved model.
"""
import pickle
import sys

import numpy as np
import tabulate

T_K = 3

fn = sys.argv[1]

with open(fn, 'rb') as fin:
    saved = pickle.load(fin)

models = saved['models']

print(saved['params'])

# Active vs. Ignition
large_vs_small = []
large_vs_small_p = []

large = models[T_K - 1][0].afm.model.large_model.fit_result
large_vs_small.append(['Large'] + list(large.params))
large_vs_small_p.append(['Large'] + list(np.round(large.pvalues, 4)))

try:
    ext = models[0][0].afm.model.fit_result_inflated
    active_vs_ign.append(['Extinction'] + list(ext.params))
    active_vs_ign_p.append(['Extinction'] + list(np.round(ext.pvalues, 4)))
except AttributeError as e:
    pass

try:
    small = models[T_K - 1][0].afm.model.small_model.fit_result
    large_vs_small.append(['Small'] + list(small.params))
    large_vs_small_p.append(['Small'] + list(np.round(small.pvalues)))
except AttributeError as e:
    pass

header = ['Model'] + list(large.params.index)
for i, h in enumerate(header):
    if h == 'np.log(np.maximum(num_det, 0.500000))':
        header[i] = 'num_det'
    if h == 'np.log(np.maximum(num_det_expon, 0.500000))':
        header[i] = 'num_det_expon'

FMT = None
print(tabulate.tabulate(large_vs_small, headers=header, tablefmt=FMT))
print(tabulate.tabulate(large_vs_small_p, headers=header, tablefmt=FMT))
print()

# Compare forecast horizon
forecast_horizon = []
forecast_horizon_p = []
model = None
for k in range(5):
    print('K==%d' % (k + 1))
    model = models[k]

    active = model[0].afm.model.fit_result_positive
    forecast_horizon.append(['(act) K==%d' % (k + 1)] + list(active.params))
    forecast_horizon_p.append(['(act) K==%d' % (k + 1)] + list(np.round(active.pvalues, 4)))

for k in range(5):
    extinction = model[0].afm.model.fit_result_inflated
    forecast_horizon.append(['(ext) K==%d' % (k + 1)] + list(extinction.params))
    forecast_horizon_p.append(['(ext) K==%d' % (k + 1)] + list(np.round(extinction.pvalues, 4)))

print(tabulate.tabulate(forecast_horizon, headers=header, tablefmt=FMT))
print(tabulate.tabulate(forecast_horizon_p, headers=header, tablefmt=FMT))
