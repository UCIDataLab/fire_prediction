from prediction.fire_clustering import cluster_over_time_with_merging
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd


def add_autoreg(df, autoreg_cols=1):
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

    for i, dct in enumerate(ind2autoregs):
        df["autoreg_%d" % (i+1)] = pd.Series(dct)
    return df


class ClusterRegression:
    """ Class for performing cluster regression for fire prediction
    """
    def __init__(self, clust_df, clust_thresh, grid_size, autoreg_max):
        """ Set up cluster model--build clusters and grid cells (if desired)
        :param modis_df: DataFrame with MODIS active fire detections
        :param gfs_dict: Dictionary with tensors of GFS weather variables
        :param clust_thresh: threshold (in KM) of where two detections are considered the same fire
        :param grid_size: size of grid on which to predict new nodes. If grid_size <= 0, don't have a grid component
        :param autoreg: max number of autoregressive columns to have
        :return:
        """
        self.clust_df = add_autoreg(clust_df, autoreg_cols=autoreg_max)
        self.clust_thresh = clust_thresh
        self.grid_size = grid_size

    def fit(self, train_years, n_autoreg, weather_vars=['temp','humidity','wind','rain']):
        """ Train on the specified years
        :param train_years: Iterable containing list of years to train on
        :return:
        """
        train_df_unmixed = self.clust_df[np.in1d(self.clust_df.year,train_years)]
        randperm = np.random.permutation(len(train_df_unmixed))
        train_df = train_df_unmixed.iloc[randperm]
        formula = "n_det ~ "
        formula += " + ".join(map(lambda x: "autoreg_%d" % x, range(1, n_autoreg+1)))
        if len(weather_vars):
            formula += " + " + " + ".join(weather_vars)
        self.fit_res = smf.glm(formula, data=train_df, family=sm.genmod.families.family.Poisson()).fit()
        return self.fit_res

    def predict(self, test_years):
        """ Get predictions for specified
        :param test_years: Iterable containing list of years to test on
        :return:
        """
