"""
Model for quantile regression.
"""
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from base.model import Model

class QuantileRegressionModel(Model):
    def __init__(self, quantile, covariates):
        """
        :param covariates: list of the names of the cols in X to use as covariates
        """
        super(QuantileRegressionModel, self).__init__()
        self.quantile = quantile
        self.covariates = covariates

        self.fit_result = None
        
    def fit(self, X, y=None):
        """
        :param y: currently unused
        """
        # Build formula for prediction
        formula = 'num_det_target ~ np.log(num_det+1)'
        if self.covariates:
            formula += ' + ' + ' + '.join(self.covariates)

        self.fit_result = smf.quantreg(formula, data=X).fit(q=self.quantile)

        return self.fit_result

    def predict(self, X):
        return self.fit_result.predict(X)
