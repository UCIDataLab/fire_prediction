"""
Model for poisson regression with zero inflation.
"""
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels import count
import statsmodels.formula.api as smf

from base.model import Model

class PoissonRegressionZeroInflatedModel(Model):
    def __init__(self, covariates):
        """
        :param covariates: list of the names of the cols in X to use as covariates
        """
        super(PoissonRegressionZeroInflatedModel, self).__init__()
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

        #self.fit_result = smf.glm(formula, data=X, family=count.PoissonZiGMLE()).fit()
        self.fit_result = count.PoissonZiGMLE(formula, data=X).fit()
        miscmodels.count.PoissonZiGMLE

        return self.fit_result

    def predict(self, X, shape=None):
        return self.fit_result.predict(X.to_dataframe())
