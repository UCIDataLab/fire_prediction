"""
Tools for evaluating predictive models.
"""
import numpy as np

def cv_years(model, X_active_df, X_ignition_c, Y_detections_c, years, t_k):
    results = []

    for test_year in years:
        train_years = [y for y in years if y!=test_year]

        # Split X_active_df
        X_active_tr_df = X_active_df[X_active_df.year.isin(train_years)]
        X_active_te_df = X_active_df[X_active_df.year==test_year]


        # Split X_ignition_c
        X_ignition_tr_c, X_ignition_te_c = X_ignition_c.remove_year(test_year)

        # Split Y_detections_c
        Y_te_c = Y_detections_c.filter_dates(X_ignition_te_c.dates[0], X_ignition_te_c.dates[-1])

        Y_te_c = Y_te_c.values

        shape = np.shape(Y_te_c)[:2]+(t_k,)
        Y_te_c = np.concatenate((Y_te_c, np.zeros(shape)), axis=2)
        Y_te_c = Y_te_c[:,:,t_k:]

        X_ignition_tr_c = X_ignition_tr_c.values
        X_ignition_te_c = X_ignition_te_c.values

        model.fit((X_active_tr_df, X_ignition_tr_c))

        Y_hat = model.predict((X_active_te_df, X_ignition_te_c))
        results.append((Y_te_c,Y_hat))

    return results
