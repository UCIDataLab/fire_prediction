"""
Model for always predicting previous day's value.
"""
import numpy as np

from .base.model import Model

class AutoregressiveGridModel(Model):
    def __init__(self):
        super(AutoregressiveGridModel, self).__init__()

        self.fit_result = None

    def fit(self, X, y=None):
        """
        :param y: currently unused
        """
        self.fit_result = None

    def predict(self, X, shape):
        pred = X['num_det'].values
        return pred
