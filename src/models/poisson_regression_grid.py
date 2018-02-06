"""
Model for fitting a bias term to each grid cell and a shared weather model with a poisson distribution.
"""
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from base.model import Model

import pandas as pd

class PoissonRegressionGridModel(Model):
    def __init__(self, covariates):
        super(PoissonRegressionGridModel, self).__init__()

        self.covariates = covariates

        self.fit_result = None

    def fit(self, X, y=None):
        """
        :param y: currently unused
        """
        """
        data = []
        for k,v in X.cubes.iteritems():
            if k in self.covariates + ['num_det', 'num_det_target']:
                data.append((k, v.values.flatten()))
        data = dict(data)

        df = pd.DataFrame(data)
        """

        formula = 'num_det_target ~ np.log(num_det+1)'
        if self.covariates:
            formula += ' + ' + ' + '.join(self.covariates)

        self.fit_result = smf.glm(formula, data=X.to_dataframe(), family=sm.genmod.families.family.Poisson()).fit()


        return self.fit_result

    def predict(self, X, shape=None):

        """
        data = []
        for k,v in X.cubes.iteritems():
            if k in self.covariates + ['num_det', 'num_det_target']:
                data.append((k, v.values.flatten()))
        data = dict(data)
        df = pd.DataFrame(data)

        pred = self.fit_result.predict(df)
        """
        pred = self.fit_result.predict(X.to_dataframe())
        #return np.reshape(pred, X.shape)
        return np.reshape(pred, shape)
