"""
Model for poisson regression with zero inflation.
"""
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels import count

from .base.model import Model


# copied from discrete_model.CountModel and adjusted
def predict(self, params, exog=None, exposure=None, offset=None,
            which='mean'):
    """
    Predict response variable of a count model given exogenous variables.

    Notes
    -----
    If exposure is specified, then it will be logged by the method.
    The user does not need to log it first.
    """
    # TODO: add offset tp
    if exog is None:
        exog = self.exog
        offset = getattr(self, 'offset', 0)
        exposure = getattr(self, 'exposure', 0)

    else:
        if exposure is None:
            exposure = 0
        else:
            exposure = np.log(exposure)
        if offset is None:
            offset = 0

    lin_pred = np.dot(exog, params[:exog.shape[1]]) + exposure + offset
    prob_poisson = 1 / (1 + np.exp(-params[-1]))
    prob_zero = (1 - prob_poisson) + prob_poisson * np.exp(-np.exp(lin_pred))

    if which == 'mean':
        return prob_poisson * np.exp(lin_pred)
    elif which == 'poisson-mean':
        return np.exp(lin_pred)
    elif which == 'linear':
        return lin_pred
    elif which == 'mean-nonzero':
        return prob_poisson * np.exp(lin_pred) / (1 - prob_zero)
    elif which == 'prob-zero':
        return prob_zero

    else:
        raise ValueError('keyword `which` not recognized')


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
        :param X: covariate dataframe
        :param y: currently unused
        """
        # Build formula for prediction
        formula = 'num_det_target ~ np.log(num_det+1)'
        if self.covariates:
            formula += ' + ' + ' + '.join(self.covariates)

        # self.fit_result = smf.glm(formula, data=X, family=count.PoissonZiGMLE()).fit()
        num_det = np.expand_dims(np.log(X['num_det'].values.flatten() + 1), 1)
        if self.covariates:
            exog = sm.add_constant(X[self.covariates].to_dataframe().as_matrix())
            exog = np.hstack([exog, num_det])
        else:
            exog = sm.add_constant(num_det)
        endog = X['num_det_target'].to_dataframe().as_matrix()

        count.PoissonZiGMLE.predict = predict
        self.fit_result = count.PoissonZiGMLE(endog, exog).fit()
        # self.fit_result = sm.GLM(endog, exog, family=sm.genmod.families.family.Poisson()).fit()
        # self.fit_result = MLPRegressor(hidden_layer_sizes=(100,50)).fit(exog, endog)

        return self.fit_result

    def predict(self, X, shape=None):
        # return self.fit_result.predict(X.to_dataframe())
        num_det = np.expand_dims(np.log(X['num_det'].values.flatten() + 1), 1)
        if self.covariates:
            exog = sm.add_constant(X[self.covariates].to_dataframe().as_matrix())
            exog = np.hstack([exog, num_det])
        else:
            exog = sm.add_constant(num_det)
        return self.fit_result.predict(exog)
