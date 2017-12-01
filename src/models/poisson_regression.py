"""
Model for poisson regression.
"""
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from base.model import Model

class PoissonRegressionModel(Model):
    def __init__(self, covariates):
        """
        :param covariates: list of the names of the cols in X to use as covariates
        """
        super(PoissonRegressionModel, self).__init__()
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

        self.fit_result = smf.glm(formula, data=X, family=sm.genmod.families.family.Poisson()).fit()
        return self.fit_result

    def predict(self, X):
        return self.fit_result.predict(X)
