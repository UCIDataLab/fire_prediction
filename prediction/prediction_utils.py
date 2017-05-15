import numpy as np
import pandas as pd
import statsmodels.api as sm


def create_dataset(df, normalize_feats=True):
    years = df.year.unique()
    X = pd.DataFrame()
    y = np.zeros((0))
    for year in years:
        annual_df = df[df.year == year]
        max_day = np.max(annual_df.dayofyear)
        min_day = np.min(annual_df.dayofyear)
        X = pd.concat((X, annual_df[annual_df.dayofyear != max_day]))
        y = np.concatenate((y, np.array(annual_df.n_det[annual_df.dayofyear != min_day])))
    if normalize_feats:
        X = (X - X.mean()) / X.std()
    return X,y


def train_test_split(df, years_in_test=1, normalize_feats=True, feat_cols=['dayofyear', 'n_det', 'vpd']):
    years = df.year.unique()
    perm = np.random.permutation(years)
    test_years = perm[0:years_in_test]
    train_years = perm[years_in_test:]
    print "Train years: " + str(train_years)
    print "Test years: " + str(test_years)
    X_train = pd.DataFrame()
    y_train = np.zeros((0))
    for year in train_years:
        annual_df = df[df.year == year]
        max_day = np.max(annual_df.dayofyear)
        min_day = np.min(annual_df.dayofyear)
        X_train = pd.concat((X_train, annual_df.loc[(annual_df.dayofyear != max_day), feat_cols]))
        y_train = np.concatenate((y_train, np.array(annual_df.n_det[annual_df.dayofyear != min_day])))
    if normalize_feats:
        X_mean = X_train.mean()
        X_std = X_train.std()
        X_train = (X_train - X_mean) / X_std
    X_train = sm.add_constant(X_train)

    X_test = pd.DataFrame()
    y_test = np.zeros((0))
    y_hat_base = np.zeros((0))
    for year in test_years:
        annual_df = df[df.year == year]
        max_day = np.max(annual_df.dayofyear)
        min_day = np.min(annual_df.dayofyear)
        X_test = pd.concat((X_test, annual_df.loc[(annual_df.dayofyear != max_day), feat_cols]))
        y_test = np.concatenate((y_test, np.array(annual_df.n_det[annual_df.dayofyear != min_day])))
        y_hat_base = np.concatenate((y_hat_base, np.array(annual_df.n_det[annual_df.dayofyear != max_day])))
    if normalize_feats:
        X_test = (X_test - X_mean) / X_std
    X_test = sm.add_constant(X_test)

    return X_train, y_train, X_test, y_test, y_hat_base