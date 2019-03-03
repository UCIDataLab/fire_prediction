import numpy as np
from cluster_regression import ClusterRegression


def all_except_year(year, rng=(2007, 2016)):
    return [x for x in range(rng[0], rng[1] + 1) if x != year]


def kill_nanners(df, weather_vars):
    for cov in weather_vars:
        nanners = df[np.isnan(df[cov])]
        for name in nanners.index:
            clust = nanners.loc[name].cluster
            dayofyear = nanners.loc[name].dayofyear
            next_offset_to_try = -1
            while 1:
                # If offset is getting too far away, just replace with mean across all time
                if abs(next_offset_to_try) > 5:
                    df.set_value(name, cov, np.mean(df[cov]))
                    break
                pot_val = df[(df.dayofyear == (dayofyear + next_offset_to_try)) & (df.cluster == clust)][cov]
                if not len(pot_val) or np.isnan(float(pot_val)):
                    if next_offset_to_try < 0:
                        next_offset_to_try = - next_offset_to_try
                    else:
                        next_offset_to_try = - (next_offset_to_try + 1)
                else:
                    df.set_value(name, cov, float(pot_val))
                    break
    return df


def cross_validation_evaluation(df, autoreg, t_k, weather_vars, zero_padding,
                                metrics=("MSE", "RobustMSE", "MeanAbsErr"),
                                return_arrs=False, kill_nanners=False, max_t_k=None,
                                legit_series=None):
    cr = ClusterRegression(df, '', '', zero_padding)
    min_year = int(np.min(df.year))
    max_year = int(np.max(df.year))

    y = []
    y_hat = []
    for year in xrange(min_year, max_year + 1):
        ft = cr.fit(all_except_year(year), autoreg, t_k, weather_vars=weather_vars, standardize_covs=False,
                    padded_beginning=zero_padding, max_t_k=max_t_k, legit_series=legit_series)

        test_df = df[df.year == year]
        if kill_nanners:
            for cov in weather_vars:
                nanners = test_df[np.isnan(test_df[cov])]
                for name in nanners.index:
                    clust = nanners.loc[name].cluster
                    dayofyear = nanners.loc[name].dayofyear
                    next_offset_to_try = -1
                    while 1:
                        # If offset is getting too far away, just replace with mean across all time
                        if abs(next_offset_to_try) > 5:
                            test_df.set_value(name, cov, np.mean(test_df[cov]))
                            break
                        pot_val = \
                            test_df[
                                (test_df.dayofyear == (dayofyear + next_offset_to_try)) & (test_df.cluster == clust)][
                                cov]
                        if not len(pot_val) or np.isnan(float(pot_val)):
                            if next_offset_to_try < 0:
                                next_offset_to_try = - next_offset_to_try
                            else:
                                next_offset_to_try = - (next_offset_to_try + 1)
                        else:
                            test_df.set_value(name, cov, float(pot_val))
                            break

        y_year = list(test_df.n_det)
        y_hat_year = list(ft.predict(test_df))
        y += y_year
        y_hat += y_hat_year

    y = np.array(y)
    y_hat = np.array(y_hat)
    ret_mets = []
    for met in metrics:
        ret_mets.append(evaluate_glm(y, y_hat, metric=met, toss_outliers=.01 * len(y)))
    if return_arrs:
        return ret_mets, y, y_hat
    else:
        return ret_mets


def evaluate_glm(y, y_hat, lins=None, ignore_nans=True, log=False, verbose=False, metric='MSE', toss_outliers=10):
    if y.shape != y_hat.shape:
        raise ValueError("y (%d) is not the same shape as y_hat (%d)" % (y.shape[0], y_hat.shape[0]))
    if ignore_nans:
        old_shape = len(y)
        non_nans = np.logical_and((1 - np.isnan(y_hat)).astype(bool), (1 - np.isnan(y)).astype(bool))
        y = y[non_nans]
        y_hat = y_hat[non_nans]
        if verbose:
            print
            "skipped %d" % (old_shape - np.sum(non_nans))
    if metric == 'MSE':
        return np.mean((y - y_hat) ** 2)
    elif metric == 'MedianSE':
        return np.median((y - y_hat) ** 2)
    elif metric == 'MeanAbsErr':
        return np.mean(np.abs(y - y_hat))
    elif metric == 'RobustMSE':
        SEs = (y - y_hat) ** 2
        sortedSEs = np.sort(SEs)
        return np.mean(sortedSEs[toss_outliers:-toss_outliers])
    elif metric == 'logMSE':
        return np.mean(np.log((y - y_hat) ** 2 + 1))
    elif metric == 'll':
        return np.mean(y * lins - np.exp(lins))
    else:
        raise ValueError(
            "Invalid metric %s. Must be one of 'MSE', 'MedianSE', 'MeanAbsErr', 'RobustMSE', or 'logMSE" % metric)
