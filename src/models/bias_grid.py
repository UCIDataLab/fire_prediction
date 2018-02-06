"""
Model for fitting a bias term to each grid cell.
"""
import numpy as np

from base.model import Model

class BiasGridModel(Model):
    def __init__(self):
        super(BiasGridModel, self).__init__()

        self.fit_result = None

    def fit(self, X, y=None):
        """
        :param y: currently unused
        """
        self.fit_result = np.expand_dims(np.mean(X, axis=2), axis=2)

    def predict(self, X):
        return self.fit_result
