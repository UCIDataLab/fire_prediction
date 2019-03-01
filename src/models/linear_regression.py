"""
Model for linear regression.
"""
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .base.model import Model
from helper import date_util as du

class LinearRegressionModel(Model):
    def __init__(self, covariates):
        """
        :param covariates: list of the names of the cols in X to use as covariates
        """
        super(LinearRegressionModel, self).__init__()
        self.covariates = covariates

        self.fit_result = None
        
    def fit(self, X, y=None):
        """
        :param y: currently unused
        """
        # Build formula for prediction
        formula = 'num_det_target ~ num_det'
        if self.covariates:
            formula += ' + ' + ' + '.join(self.covariates)

        #self.fit_result = smf.glm(formula, data=X, family=sm.genmod.families.family.Gaussian()).fit()
        self.fit_result = smf.ols(formula=formula, data=X).fit()
        return self.fit_result

    def predict(self, X):
        return self.fit_result.predict(X)
