"""
Tools for evaluating predictive models.
"""
import numpy as np

def cross_validation_years(model, X):
    results = []

    years = range(int(X.year.min()), int(X.year.max())+1)
    for test_year in years:
        train_years = [y for y in years if y!=test_year]

        X_train = X[X.year.isin(train_years)]

        X_test = X[X.year==test_year]
        y_test = np.array(X_test.num_det)

        model.fit(X_train)

        y_hat = np.array(model.predict(X_test))
        results.append((y_test, y_hat))

    return results, years

def leave_none_out(model, X):
    results = []

    model.fit(X)

    y_hat = np.array(model.predict(X))
    results.append((X.num_det, y_hat))

    return results, X.year.unique()

