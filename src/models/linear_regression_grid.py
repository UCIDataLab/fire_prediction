"""
Model for fitting a bias term to each grid cell and a shared weather model with a linear distribution.
"""
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from StringIO import StringIO

from base.model import Model
from sklearn.neural_network import MLPRegressor

import pandas as pd

class LinearRegressionGridModel(Model):
    def __init__(self, covariates, regularizer_weight=None, filter_func=None, pred_func=None):
        super(LinearRegressionGridModel, self).__init__()

        self.covariates = covariates
        self.regularizer_weight = regularizer_weight
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

        formula = 'num_det_target ~ np.log(num_det+1)'
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
            #self.fit_result = smf.glm(formula, data=X_df, family=sm.genmod.families.family.Poisson()).fit()
            self.fit_result = smf.ols(formula=formula, data=X_df).fit()
        else:
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

        pred = self.fit_result.predict(X_df)

        if self.pred_func:
            pred = self.pred_func(X_df, pred)

        pred = np.reshape(pred, shape, order='F')

        return pred
