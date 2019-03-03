"""
Model for predicting all detections in a region by summing over cluster predictions.
"""
import datetime as dtime

import numpy as np
import pandas as pd

from src.helper import date_util as du
from src.helper import df_util as dfu
from .base.model import Model


class RegionalSummationModel(Model):
    def __init__(self, t_k, covariates, cluster_model, ignition_model=None, date_range=((5, 14), (8, 31))):
        super(RegionalSummationModel, self).__init__()

        self.t_k = t_k
        self.covariates = covariates
        self.cluster_model = cluster_model
        self.ignition_model = ignition_model
        self.date_range = date_range
        self.ignition_bias = None

    def fit(self, X, y=None):
        """
        :param X: covariate dataframe
        :param y: currently unused
        """
        self.cluster_model.fit(X)

        if self.ignition_model == 'mean':
            X_region = self.build_regional_data(X)
            y_hat = self.predict_cluster(X)
            mean_residual_target = np.mean(X_region.num_det_target - y_hat)
            self.ignition_bias = mean_residual_target
        if self.ignition_model == 'median':
            X_region = self.build_regional_data(X)
            y_hat = self.predict_cluster(X)
            median_residual_target = np.median(X_region.num_det_target - y_hat)
            self.ignition_bias = median_residual_target
        else:
            self.ignition_bias = 0

    def predict_cluster(self, X):
        y_hat_clusters = np.array(self.cluster_model.predict(X))
        X['y_hat'] = y_hat_clusters

        dates = self.get_date_range_list(X)
        y_hat = np.empty(len(dates))

        for i, day in enumerate(dates):
            y_hat_day = np.sum(X[X.date_local == day].y_hat)
            y_hat[i] = y_hat_day

        return y_hat

    def predict(self, X, shape=None):
        return self.predict_cluster(X) + self.ignition_bias

    def get_date_range_list(self, X):
        year_range = dfu.get_year_range(X, 'date_local')

        dates = []
        for year in range(year_range[0], year_range[1] + 1):
            begin = dtime.date(year, self.date_range[0][0], self.date_range[0][1])
            end = dtime.date(year, self.date_range[1][0], self.date_range[1][1])

            dates += list(du.date_range(begin, end + du.INC_ONE_DAY))

        return dates

    def build_regional_data(self, X):
        dates = self.get_date_range_list(X)
        daily_det = np.empty(len(dates))
        daily_det_target = np.empty(len(dates))

        for i, date in enumerate(dates):
            daily_det[i] = np.sum(X[X.date_local == date].num_det)
            daily_det_target[i] = np.sum(X[X.date_local == (date + du.INC_ONE_DAY * self.t_k)].num_det)

        X_region = pd.DataFrame(np.array([dates, daily_det, daily_det_target]).T,
                                columns=['date_local', 'num_det', 'num_det_target'])

        return X_region

    @staticmethod
    def preprocess_data(X):
        return X
