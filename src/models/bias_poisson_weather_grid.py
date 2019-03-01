"""
Model for fitting a bias term to each grid cell and a shared weather model with a poisson distribution.
"""
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .base.model import Model

import pandas as pd

class BiasPoissonWeatherGridModel(Model):
    def __init__(self, covariates):
        super(BiasPoissonWeatherGridModel, self).__init__()

        self.covariates = covariates

        #self.fit_bias = None
        self.fit_result = None

    def fit(self, X, y=None):
        """
        :param y: currently unused
        """
        # Fit bias component
        #ignition = X['ignition'].values
        #self.fit_bias = np.expand_dims(np.mean(ignition, axis=2), axis=2)

        #data = dict(map(lambda x: (x,X.cubes[x].values.flatten()), X.cubes))
        data = []
        for k,v in X.cubes.items():
            if k in self.covariates + ['num_ig_target']:
                data.append((k, v.values.flatten()))
        data = dict(data)

        df = pd.DataFrame(data)

        # Fit weather component
        #X_central = X - self.fit_bias

        formula = 'num_ig_target ~ 1'
        if self.covariates:
            formula += ' + ' + ' + '.join(self.covariates)

        self.fit_result = smf.glm(formula, data=df, family=sm.genmod.families.family.Poisson()).fit()

        return self.fit_result

    def predict(self, X):

        data = []
        for k,v in X.cubes.items():
            if k in self.covariates + ['num_ig_target']:
                data.append((k, v.values.flatten()))
        data = dict(data)
        df = pd.DataFrame(data)

        pred = self.fit_result.predict(df) #+ self.fit_bias
        return np.reshape(pred, X.shape)
