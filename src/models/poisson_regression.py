"""
Model for poisson regression.
"""
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from base.model import Model
from helper import date_util as du

class PoissonRegressionModel(Model):
    def __init__(self, t_k, covariates):
        """
        :param t_k: num of days ahead to predict (0: today, 1: tomorrow, ...)
        :param covariates: list of the names of the cols in X to use as covariates
        """
        super(PoissonRegressionModel, self).__init__()
        self.t_k = t_k
        self.covariates = covariates

        self.fit_result = None
        
    def fit(self, X, y=None):
        """
        :param y: currently unused
        """
        #X = self.preprocess_data(X)
        #X = X.iloc[np.random.permutation(X.shape[0])]

        # Build formula for prediction
        formula = 'num_det_target ~ np.log(num_det+1)'
        if self.covariates:
            formula += ' + ' + ' + '.join(self.covariates)

        self.fit_result = smf.glm(formula, data=X, family=sm.genmod.families.family.Poisson()).fit()
        return self.fit_result

    def predict(self, X):
        return self.fit_result.predict(X)

    def preprocess_data(self, X):
        """
        Prepare data for the model.
        """
        X = self.add_autoregressive_col(X, self.t_k+1)
        X = X.iloc[np.random.permutation(X.shape[0])]

        return X

    def add_autoregressive_col(self, X, t_offset):
        """
        Add an autoregressive column to the data.

        :param t_offset: num of days ahead to generate col from (1: tomorrow, ...)
        """
        num_det_target = np.empty(X.shape[0])
        for i, row in enumerate(X.itertuples()):
            date, cluster_id, num_det = row.date_local, row.cluster_id, row.num_det

            cluster_df = X[(X.cluster_id==cluster_id) & (X.date_local==date+du.INC_ONE_DAY*(t_offset))]
            val = cluster_df.num_det.iloc[0] if not cluster_df.empty else 0

            num_det_target[i] = val

        return X.assign(num_det_target=num_det_target)

