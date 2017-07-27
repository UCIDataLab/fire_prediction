import numpy as np
import statsmodels.api as sm
import pandas as pd
from scipy.misc import factorial


def get_regression_df(old_global_df, covar_cols=['temp', 'vpd'], normalize=[1,1], log_counts=True, autocorr_windows=[1],
                      ignore_nans=True, return_alt_ys=False):
    """ Create a DataFrame and output vector in a format amenable to statsmodels.
    :param global_df: A DataFrame with a column of counts called 'n_det', 'year' column, 'dayofyear' column,
                        and other covariate columns
    :param covar_cols: Names of covariate columns to use for prediction
    :param normalize: For each covariate column, do we want to use as is (False) or subtract mean and divide by variance (True)
    :param log_counts: Does our output 'n_det' column feature counts (False) or log counts (True)
    :param autocorr_window: the n_det column in the output DataFrame will be an average of the past autocorr_window days
    :return: A DataFrame with len(feat_cols) columns, an output vector, where the ith row of the DataFrame should
                predict the ith row of the output vector, and a vector with the day/year of the output to keep track
    """
    if ignore_nans:
        new_df = pd.DataFrame()
        for col in old_global_df.columns:
            new_df[col] = pd.Series()
        for row in old_global_df.index:
            if all(np.logical_not(np.isnan(old_global_df.loc[row]))):
                row_to_add = old_global_df.loc[row]
                new_df.loc[row] = row_to_add
        print "Old df had %d rows, new one without nans has %d" % (len(old_global_df), len(new_df))
        global_df = new_df
    else:
        global_df = old_global_df
    X = pd.DataFrame()
    y = []
    y_dates = []
    for w in autocorr_windows:
        X['n_det_%d' % w] = pd.Series()
    new_covars = []
    for col, norm in zip(covar_cols, normalize):
        if norm:
            new_covars.append('norm' + col)
            global_df['norm' + col] = (global_df[col] - np.mean(global_df[col])) / np.std(global_df[col])
            X['norm' + col] = pd.Series()
        if norm == 2 or norm == 0:
            new_covars.append(col)
            X[col] = pd.Series()
    covar_cols = new_covars

    years = global_df.year.unique()
    for year in years:
        min_day = np.min(global_df.dayofyear)
        max_day = np.max(global_df.dayofyear)
        y_days = np.arange(min_day+max(autocorr_windows), max_day+1)
        for day in y_days:
            row_dict = dict()
            # Get the covariates for the result day
            try:
                covar_vals = global_df.loc[lambda x: (x.year==year) & (x.dayofyear==day), covar_cols]
                y.append(int(global_df.loc[lambda x: (x.year==year) & (x.dayofyear==day), 'n_det']))
                y_dates.append((day, year))
            except TypeError:
                continue
            for col in covar_cols:
                val = float(covar_vals[col])
                row_dict[col] = val
            # Get the past autocorr_window days
            prev_days = np.arange(day-max(autocorr_windows), day)
            prev_counts = []
            for day in prev_days:
                try:
                    count = global_df.loc[lambda x: (x.year==year) & (x.dayofyear==day), 'n_det']
                    prev_counts.append(int(count))
                except KeyError:
                    prev_counts.append(0)
                except TypeError:
                    if len(global_df.loc[lambda x: (x.year==year) & (x.dayofyear==day), 'n_det']):
                        print "nopenopenope"
                        raise ValueError
                    else:
                        prev_counts.append(0)
            for w in autocorr_windows:
                if log_counts:
                    row_dict['n_det_%d' % w] = np.log(np.mean(prev_counts[-w])+1)
                else:
                    row_dict['n_det_%d' % w] = np.mean(prev_counts[-w])

            # Now put stuff into the DataFrame
            X.loc[len(y)-1] = row_dict

    y = np.array(y)
    X.loc[:,'y'] = y
    if return_alt_ys:
        bin_y = y != 0
        nz_y = y[bin_y]
        nz_X = X.loc[bin_y]
        X.loc[:,'bin_y'] = bin_y.astype(int)
        return X, y, nz_X, nz_y, bin_y
    return X, y, y_dates


def train_test_split(X, y, y_dates=None, test_perc=.2, valid_perc=0., idx=None):
    """ Split X and y into train and test
    :param X: input DataFrame
    :param y: output vector
    :param y_dates: (optionally) a list of when each y happens for bookkeeping
    :param test_perc: percentage of data to put in the testing set
    :param valid_perc: percentage of data to put in the validation set (could be 0)
    :return: X_train, y_train, (y_dates_train), X_test, y_test, (y_dates_test), (X_valid), (y_valid), (y_dates_valid)
    """
    N = len(X)
    if idx is not None:
        perm = idx
    else:
        perm = np.random.permutation(N)
    train_cutoff = int(N*(1 - (test_perc+valid_perc)))
    train_inds = perm[0:train_cutoff]
    if valid_perc:
        test_cutoff = int(N*(1-valid_perc))
        test_inds = perm[train_cutoff:test_cutoff]
        valid_inds = perm[test_cutoff:]
    else:
        test_inds = perm[train_cutoff:]

    ret_list = []
    ret_list.append(X.loc[train_inds])  # X_train
    ret_list.append(y[train_inds])  # y_train
    if y_dates:
        ret_list.append([y_dates[i] for i in train_inds])  # y_dates_train

    ret_list.append(X.loc[test_inds])  # X_test
    ret_list.append(y[test_inds])  # y_test
    if y_dates:
        ret_list.append([y_dates[i] for i in test_inds])

    if valid_perc:
        ret_list.append(X.loc[valid_inds])  # X_valid
        ret_list.append(y[valid_inds])  # y_valid
        if y_dates:
            ret_list.append([y_dates[i] for i in valid_inds])

    return tuple(ret_list)


def get_glm(X_train, y_train):
    glm = sm.GLM(y_train, X_train, family=sm.genmod.families.family.Poisson(), missing='drop')
    glm_res = glm.fit()
    return glm_res


def evaluate_glm(y, y_hat, lins=None, ignore_nans=True, log=False, verbose=False, metric='MSE', toss_outliers=10):

    if y.shape != y_hat.shape:
        raise ValueError("y (%d) is not the same shape as y_hat (%d)" % (y.shape[0], y_hat.shape[0]))
    if ignore_nans:
        non_nans = (1 - np.isnan(y_hat)).astype(bool)
        y = y[non_nans]
        y_hat = y_hat[non_nans]
        if verbose:
            print "skipped %d" %(len(y) - np.sum(non_nans))

    if metric == 'MSE':
        return np.mean((y - y_hat)**2)
    elif metric == 'MedianSE':
        return np.median((y - y_hat)**2)
    elif metric == 'MeanAbsErr':
        return np.mean(np.abs(y - y_hat))
    elif metric == 'RobustMSE':
        SEs = (y - y_hat)**2
        sortedSEs = np.sort(SEs)
        return np.mean(sortedSEs[toss_outliers:-toss_outliers])
    elif metric == 'logMSE':
        return np.mean(np.log((y - y_hat)**2+1))
    elif metric == 'll':
        return np.mean(y * lins - np.exp(lins))
    else:
        raise ValueError("Invalid metric. Must be one of 'MSE', 'MedianSE', 'MeanAbsErr', 'RobustMSE', or 'logMSE")
