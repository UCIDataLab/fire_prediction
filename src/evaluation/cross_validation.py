"""
Tools for evaluating predictive models.
"""
import numpy as np

import pandas as pd

def cv_years(model, X, y, years, t_k):
    results_te = []
    results_tr = []
    models = []

    for test_year in years:
        # Split X
        X_tr, X_te = X.remove_year(test_year)

        # Split Y
        y_tr = y.sel(time=y.time.dt.year != test_year).values
        y_te = y.sel(time=y.time.dt.year == test_year).values

        fit_model = model.fit(X_tr,y_tr)

        y_hat_tr = model.predict(X_tr, np.shape(y_tr))
        y_hat_te = model.predict(X_te, np.shape(y_te))

        # Store results
        results_tr.append((y_tr,y_hat_tr))
        results_te.append((y_te,y_hat_te))
        models.append(fit_model)

    return (results_tr, results_te), models

def evaluate_all(model, X, y, t_k):
    results_te = []
    results_tr = []
    models = []

    # Split X
    X_tr, X_te = X, X

    # Split Y
    y_tr, y_te = y.values, y.values

    fit_model = model.fit(X_tr,y_tr)

    y_hat_te = model.predict(X_te, np.shape(y_te))

    # Store results
    results_tr.append((y_tr,y_hat_te)) # Keep this just so it matches output format for cv_years
    results_te.append((y_te,y_hat_te))
    models.append(fit_model)

    return (results_tr, results_te), models

def make_year_list(years, shape):
    num = shape / len(years)
    vals = []
    for y in years:
        vals += [y]*num

    return vals
    
def add_cell_encoding(data):
    for i in range(33):
        for j in range(55):
            enc = np.zeros((33,55,1100))
            enc[i,j,:]=1
            enc = enc.flatten()
            data.append(('cell_%d_%d'%(i,j),enc))

    print [k[0] for k in data][:5]

def cv_years_grid(model, X_active_r, X_ignition_c, Y_detections_c, years, t_k):
    results = []

    data = []
    for k,v in X_active_r.cubes.iteritems():
        if k in model.afm.covariates + ['num_det', 'num_det_target']:
            data.append((k, v.values.flatten()))

    year_list = make_year_list([d.year for d in X_active_r.dates], len(data[0][1]))
    data.append(('year', year_list))

    add_cell_encoding(data)

    data = dict(data)
    X_active_df = pd.DataFrame(data)

    for test_year in years:
        train_years = [y for y in years if y!=test_year]

        # Split X_active_df
        #X_active_tr_r, X_active_te_r = X_ignition_c.remove_year(test_year)
        X_active_tr_df = X_active_df[X_active_df.year.isin(train_years)]
        X_active_te_df = X_active_df[X_active_df.year==test_year]

        # Split X_ignition_c
        X_ignition_tr_c, X_ignition_te_c = X_ignition_c.remove_year(test_year)

        # Split Y_detections_c
        Y_te_c = Y_detections_c.filter_dates(X_ignition_te_c.dates[0], X_ignition_te_c.dates[-1])

        Y_te_c = Y_te_c.values

        # Shift forwards t_k days
        shape = np.shape(Y_te_c)[:2]+(t_k,)
        Y_te_c = np.concatenate((Y_te_c, np.zeros(shape)), axis=2)
        Y_te_c = Y_te_c[:,:,t_k:]

        X_ignition_tr_c = X_ignition_tr_c.values
        X_ignition_te_c = X_ignition_te_c.values

        model.fit((X_active_tr_df, X_ignition_tr_c))

        Y_hat = model.predict((X_active_te_df, X_ignition_te_c))
        results.append((Y_te_c,Y_hat))

    return results
