import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def add_autoreg_and_n_det(df, autoreg_cols=1, t_k_max=1, zero_padding=True):
    """ Take a clust_feat_df and convert it into a form friendly to statsmodel regression
    Format: [cluster, [autoreg_coefs], [everything else already there]]
    :param: df: a DataFrame of the clust_feat_df form
    :param autoreg_cols: number of autoregressive coefficients to use
    :return: DataFrame where each row is one of the clusters in one of the years we want
    """
    # Build dicts that will ultimately become new columns in our DataFrame
    ind2autoregs = []
    for col in xrange(autoreg_cols):
        ind2autoregs.append(dict())

    ind2t_ks = []
    for col in xrange(t_k_max):
        ind2t_ks.append(dict())

    # Get mapping from merged clusters mergees to mergers and when the merge happened
    mergee2merger = dict()
    for clust in df.cluster.unique():
        if np.isnan(clust):
            continue
        # Find the merger this guy is a part of
        alt_clust = df[df.cluster==clust].alt_cluster.unique()[0]
        if isinstance(alt_clust, tuple):
            mergee = alt_clust[0]
            day = alt_clust[1]
            if mergee not in mergee2merger:
                mergee2merger[mergee] = set()
            mergee2merger[mergee].add((clust,day))

    for clust in df.cluster.unique():
        if np.isnan(clust):
            continue
        clust_df = df[df.cluster==clust].sort('dayofyear')
        alt_clust = clust_df.alt_cluster.unique()[0]
        time_series = np.array(clust_df.n_det)
        days = np.array(clust_df.dayofyear)
        for i, day in enumerate(days):
            ind = clust_df[clust_df.dayofyear==day].index[0]
            for autoreg in xrange(autoreg_cols):
                pos = i - autoreg - 1
                if pos < 0:
                    ind2autoregs[autoreg][ind] = 0.
                else:
                    ind2autoregs[autoreg][ind] = time_series[pos]
                daypos = pos + days[0]
                if clust in mergee2merger:
                    for (merger,day) in mergee2merger[clust]:
                        merge_amt = df[(df.cluster==merger) & (df.dayofyear==daypos)]
                        if len(merge_amt):
                            ind2autoregs[autoreg][ind] += int(merge_amt.n_det)
                ind2autoregs[autoreg][ind] = np.log(ind2autoregs[autoreg][ind]+1)

            for t_k in xrange(t_k_max):
                pos = i + t_k
                if pos >= len(time_series) and not isinstance(alt_clust, tuple):
                    # CASE 1: we've run out of positions and don't merge
                    # Add either 0 or NaN depending on zero padding
                    if zero_padding:
                        ind2t_ks[t_k][ind] = 0
                    else:
                        ind2t_ks[t_k][ind] = np.nan
                elif pos >= len(time_series):
                    # CASE 2: we merge into someone else. For now, just predict the merged pixels
                    mergee = alt_clust[0]
                    merger_dets = df[(df.cluster == mergee) & (df.dayofyear == pos + days[0])]
                    if len(merger_dets):   # we have detections
                        ind2t_ks[t_k][ind] = merger_dets.iloc[0].n_det
                    else:  # deal with zeros as above
                        if zero_padding:
                            ind2t_ks[t_k][ind] = 0
                        else:
                            ind2t_ks[t_k][ind] = np.nan
                else:
                    # CASE 3: We're normal
                    ind2t_ks[t_k][ind] = time_series[pos]

    for i, dct in enumerate(ind2autoregs):
        df["autoreg_%d" % (i+1)] = pd.Series(dct)
    for i,dct in enumerate(ind2t_ks):
        df["t_k_%d" % i] = pd.Series(dct)
    return df


class ClusterRegression:
    """ Class for performing cluster regression for fire prediction
    """
    def __init__(self, clust_df, clust_thresh, grid_size, zero_padding=True):
        """ Set up cluster model--build clusters and grid cells (if desired)
        :param modis_df: DataFrame with MODIS active fire detections
        :param gfs_dict: Dictionary with tensors of GFS weather variables
        :param clust_thresh: threshold (in KM) of where two detections are considered the same fire
        :param grid_size: size of grid on which to predict new nodes. If grid_size <= 0, don't have a grid component
        :return:
        """
        self.clust_df = clust_df.copy()
        self.clust_thresh = clust_thresh
        self.grid_size = grid_size
        self.zero_padding = zero_padding

    def fit(self, train_years, n_autoreg, t_k=0, weather_vars=('temp','humidity','wind','rain'),
            standardize_covs=False, padded_beginning=True, max_t_k=None, legit_series=None):
        """ Train on the specified years
        :param train_years: Iterable containing list of years to train on
        :param n_autoreg: number of autoregressive columns in the regression
        :param t_k: how many days ahead to predict (0: predict today, 1: predict tomorrow, etc)
        :param weather_vars: list of column names of weather covariates
        :param standardize_covs: if True, instead of regressing directly on the weather covariates, standardize them
                                    first by subtracting the mean and dividing by standard deviation
        :return:
        """
        #max_autoreg = n_autoreg + t_k
        #if "autoreg_%d" % max_autoreg not in self.clust_df.columns:  # or "t_k_%d" % t_k not in self.clust_df.columns:
        #    self.clust_df = add_autoreg_and_n_det(self.clust_df, n_autoreg, t_k+1, zero_padding=self.zero_padding)
        train_df_unmixed = self.clust_df[np.in1d(self.clust_df.year,train_years)]  # & (~np.isnan(self.clust_df["t_k_%d" % t_k]))]
        # Kill beginning padding if desired
        if not padded_beginning:
            if legit_series is None:
                legit_series = pd.Series(index=train_df_unmixed.index)
                for clust in train_df_unmixed.cluster.unique():
                    clust_df = train_df_unmixed[train_df_unmixed.cluster==clust]
                    legit_day = np.min(clust_df.dayofyear) + max_t_k
                    legit_series[clust_df[clust_df.dayofyear >= legit_day].index] = 1
            train_df_unmixed = train_df_unmixed[legit_series == 1]

        randperm = np.random.permutation(len(train_df_unmixed))
        train_df = train_df_unmixed.iloc[randperm]
        if standardize_covs:
            for reg in xrange(t_k + 1, t_k + n_autoreg + 1):
                col = train_df["autoreg_%d" % reg]
                train_df["autoreg_%d_stand" % reg] = (col - np.mean(col)) / np.std(col)
            for var in weather_vars:
                train_df[var + "_stand"] = (train_df[var] - np.mean(train_df[var])) / np.std(train_df[var])
            weather_vars = map(lambda x: x + "_stand", weather_vars)
        formula = "n_det ~ "
        if standardize_covs:
            formula += " + ".join(map(lambda x: "autoreg_%d_stand" % x, range(t_k + 1, t_k + n_autoreg + 1)))
        else:
            formula += " + ".join(map(lambda x: "autoreg_%d" % x, range(t_k + 1, t_k + n_autoreg + 1)))
        if len(weather_vars):
            formula += " + " + " + ".join(weather_vars)
        self.fit_res = smf.glm(formula, data=train_df, family=sm.genmod.families.family.Poisson()).fit()
        return self.fit_res

    def predict(self, test_years):
        """ Get predictions for specified
        :param test_years: Iterable containing list of years to test on
        :return:
        """
