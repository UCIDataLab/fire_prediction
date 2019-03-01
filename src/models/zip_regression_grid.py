"""
Model for fitting a bias term to each grid cell and a shared weather model with a zero-inflated poisson distribution.
"""
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP

from io import StringIO

from .base.model import Model
from sklearn.neural_network import MLPRegressor

import pandas as pd

class ZIPRegressionGridModel(Model):
    def __init__(self, covariates, regularizer_weight=None, log_shift=1, log_correction='add', filter_func=None, pred_func=None):
        super(ZIPRegressionGridModel, self).__init__()

        self.covariates = covariates
        self.regularizer_weight = regularizer_weight
        self.log_shift = log_shift
        self.log_correction = log_correction
        self.filter_func = filter_func
        self.pred_func = pred_func

        self.fit_result = None

    def fit(self, X, y=None):
        """
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
        elif self.log_correction == 'max':
            formula = 'num_det_target ~ np.log(np.maximum(%f,num_det))' % self.log_shift
        else:
            raise ValueError('Invalid log_correction: %s' % self.log_correction)
        if self.covariates:
            formula += ' + ' + ' + '.join(self.covariates)

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

        #print(pd.DataFrame.from_csv(StringIO(X_df.to_csv())))
        X_df = pd.DataFrame.from_csv(StringIO(X_df.to_csv()))
        #print(X_df)

        if self.regularizer_weight is None:
            #self.fit_result = ZeroInflatedPoisson.from_formula(formula, data=X_df).fit()
            formula = 'num_det_target ~ np.log(num_det+%f)' % self.log_shift

            y = np.zeros(len(X_df))
            y[:] = X_df['num_det_target']

            data = np.zeros((len(X_df), 2 + len(self.covariates)))
            data[:,1] = np.log(X_df['num_det'] + 1)
            for i,cov in enumerate(self.covariates):
                data[:,i+2] = X_df[cov]

            self.mean = np.mean(data, axis=0)
            self.std = np.ones(len(self.covariates)+2)#np.std(data, axis=0)
            data = (data-self.mean)/self.std

            data[:,0] = 1

            self.fit_result = ZeroInflatedPoisson(y, exog=data,exog_infl=data).fit()
            #smf.glm(formula, data=X_df, family=sm.genmod.families.family.Poisson()).fit()
        else:
            #self.fit_result = ZeroInflatedPoisson.from_formula(formula, data=X_df).fit_regularized(alpha=self.regularizer_weight)
            raise NotImplementedError()
        #self.fit_result = MLPRegressor(hidden_layer_sizes=(100,50)).fit(exog, endog)


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

        data = np.zeros((len(X_df), 2 + len(self.covariates)))
        data[:,1] = np.log(X_df['num_det'] + 1)
        for i,cov in enumerate(self.covariates):
            data[:,i+2] = X_df[cov]

        data = (data-self.mean)/self.std
        data[:,0] = 1

        pred = self.fit_result.predict(exog=data, exog_infl=data)

        if self.pred_func:
            pred = self.pred_func(X_df, pred)

        pred = np.array(pred)
        pred = np.reshape(pred, shape, order='F')

        return pred
