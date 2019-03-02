"""
Model for always predicting previous day's value.
"""

from .base.model import Model


class AutoregressiveGridModel(Model):
    def __init__(self):
        super(AutoregressiveGridModel, self).__init__()

        self.fit_result = None

    def fit(self, X, y=None):
        """
        :param X: covariate dataframe
        :param y: currently unused
        """
        self.fit_result = None

    def predict(self, X, shape=None):
        pred = X['num_det'].values
        return pred
