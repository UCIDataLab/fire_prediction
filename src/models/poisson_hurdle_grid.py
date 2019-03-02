"""
Model for fitting a bias term to each grid cell and a shared weather model with a poisson distribution.
"""
from io import StringIO

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as sd
import statsmodels.formula.api as smf

from .base.model import Model


class PoissonRegressionHurdleGridModel(Model):
    def __init__(self, covariates, regularizer_weight=None, log_shift=1, log_correction='add', filter_func=None,
                 pred_func=None):
        super().__init__()
        self.covariates = covariates
        self.regularizer_weight = regularizer_weight
        self.log_shift = log_shift
        self.log_correction = log_correction
        self.filter_func = filter_func
        self.pred_func = pred_func

        self.fit_result = None
        self.fit_ignition = None

    def fit(self, X, y=None):
        """
        :param X: covariate dataframe
        :param y: currently unused
        """
        """
        data = []
        for k,v in X.cubes.items():
            if k in self.covariates + ['num_det', 'num_det_target']:
                data.append((k, v.values.flatten()))
        data = dict(data)

        df = pd.DataFrame(data)
        """

        if self.log_correction == 'add':
            formula = 'num_det_target ~ np.log(num_det+%f)' % self.log_shift
            formula_ign = 'active ~ np.log(num_det+%f)' % self.log_shift
        elif self.log_correction == 'max':
            formula = 'num_det_target ~ np.log(np.maximum(%f,num_det))' % self.log_shift
            formula_ign = 'active ~ np.log(np.maximum(%f,num_det))' % self.log_shift
        else:
            raise ValueError('Invalid log_correction: %s' % self.log_correction)
        if self.covariates:
            formula += ' + ' + ' + '.join(self.covariates)
            formula_ign += ' + ' + ' + '.join(self.covariates)

        """
        num_det = np.expand_dims(np.log(X['num_det'].values.flatten()+1),1)
        if self.covariates:
            exog = sm.add_constant(X[self.covariates].to_dataframe().as_matrix())
            exog = np.hstack([exog,num_det])
        else:
            exog = sm.add_constant(num_det)
        endog = X['num_det_target'].to_dataframe().as_matrix()
        """

        X_df = X.to_dataframe()
        if self.filter_func:
            X_df = self.filter_func(X_df)

        # print(pd.DataFrame.from_csv(StringIO(X_df.to_csv())))
        X_df = pd.DataFrame.from_csv(StringIO(X_df.to_csv()))
        # print(X_df)

        self.fit_ignition = sd.Logit.from_formula(formula, data=X_df).fit()

        if self.regularizer_weight is None:
            self.fit_result = smf.glm(formula, data=X_df[X_df.num_det_target != 0],
                                      family=sm.genmod.families.family.Poisson()).fit()
        else:
            self.fit_result = smf.glm(formula, data=X_df[X_df.num_det_target != 0],
                                      family=sm.genmod.families.family.Poisson()).fit_regularized(
                alpha=self.regularizer_weight)
        # self.fit_result = MLPRegressor(hidden_layer_sizes=(100,50)).fit(exog, endog)

        return self.fit_result

    def predict(self, X, shape=None):

        """
        data = []
        for k,v in X.cubes.items():
            if k in self.covariates + ['num_det', 'num_det_target']:
                data.append((k, v.values.flatten()))
        data = dict(data)
        df = pd.DataFrame(data)

        pred = self.fit_result.predict(df)
        """
        """
        num_det = np.expand_dims(np.log(X['num_det'].values.flatten()+1),1)
        if self.covariates:
            exog = sm.add_constant(X[self.covariates].to_dataframe().as_matrix())
            exog = np.hstack([exog,num_det])
        else:
            exog = sm.add_constant(num_det)
 
        pred = self.fit_result.predict(exog)
        """

        X_df = X.to_dataframe()

        pred = self.fit_result.predict(X_df)
        pred2 = self.fit_ignition.predict(X_df)

        pred = pred * pred2

        if self.pred_func:
            pred = self.pred_func(X_df, pred)

        pred = np.array(pred)
        pred = np.reshape(pred, shape, order='F')

        return pred
